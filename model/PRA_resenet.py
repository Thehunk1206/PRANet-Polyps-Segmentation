'''
MIT License

Copyright (c) 2020 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from model.ra_module import ReverseAttention
from model.partial_decoder import PartialDecoder
from model.rfb import RFB
from model.backbone import FE_backbone


class PRAresnet(tf.keras.Model):
    def __init__(self, IMG_H: int = 352, IMG_W: int = 352, filters: int = 32, **kwargs):
        super(PRAresnet, self).__init__(**kwargs)
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.filters = filters

        # pretrained resnet
        self.fe_backbone = FE_backbone(inshape=(self.IMG_H, self.IMG_W, 3))
        self.resnet = self.fe_backbone.get_fe_backbone()

        # Receptive field blocks
        # 3 blocks for three high level features from resnet
        self.rfb_2 = RFB(filters=self.filters, name="rfb_2")
        self.rfb_3 = RFB(filters=self.filters, name="rfb_3")
        self.rfb_4 = RFB(filters=self.filters, name="rfb_4")

        # Paraller Partial Decoder Block
        self.ppd = PartialDecoder(filters=self.filters, name="partial_decoder")
        self.resize_sg = preprocessing.Resizing(self.IMG_H, self.IMG_W, name='salient_out_5')

        # reverse attention branch 4
        self.resize_4 = preprocessing.Resizing(self.IMG_H//32, self.IMG_W//32, name="resize4")
        self.ra_4 = ReverseAttention(
            filters=256, kernel_size=(5, 5), name="reverse_attention_br4")
        self.resize_s4 = preprocessing.Resizing(self.IMG_H, self.IMG_W,name="salient_out_4")

        # reverse attention branch 3
        self.resize_3 = preprocessing.Resizing(self.IMG_H//16, self.IMG_W//16, name="resize3")
        self.ra_3 = ReverseAttention(filters=64, name="reverse_attention_br3")
        self.resize_s3 = preprocessing.Resizing(self.IMG_H, self.IMG_W, name="salient_out_3")

        # reverse attention branch 2
        self.resize_2 = preprocessing.Resizing(self.IMG_H//8, self.IMG_W//8, name="resize2")
        self.ra_2 = ReverseAttention(filters=64, name="reverse_attention_br2")
        self.resize_s2 = preprocessing.Resizing(self.IMG_H, self.IMG_W, name="final_salient_out_2")


    def call(self, x: tf.Tensor):
        self.features = self.resnet(x)

        # RFB
        feat2_rfb = self.rfb_2(self.features[1])  # => level_2(batch,h/8,w/8,32)
        feat3_rfb = self.rfb_3(self.features[2])  # => level_3(batch,h/16,w/16,32)
        feat4_rfb = self.rfb_4(self.features[3])  # => level_4(batch,h/32,w/32,32)

        # Partial decoder 
        sg = self.ppd(feat4_rfb, feat3_rfb, feat2_rfb) # => (batch,h/8,w/8,1) Global saliency map
        lateral_out_sg = self.resize_sg(sg)# resize (batch,h/8,w/8,1) => (batch,h,w,1) #out 5

        # reverse attention branch 4 
        resized_sg = self.resize_4(sg)# resize (batch, h/8,w/8,1) => (batch, h/32,w/32,1)
        s4 = self.ra_4(self.features[3],resized_sg)
        lateral_out_s4 = self.resize_s4(s4)# resize (batch,h/32,w/32,1) => (batch,h,w,1) #out 4


        # reverse attention branch 3
        resized_s4 = self.resize_3(s4)# resize (batch, h/32,w/32,1) => (batch, h/16,w/16,1)
        s3 = self.ra_3(self.features[2],resized_s4)
        lateral_out_s3 = self.resize_s3(s3)# resize (batch,h/16,w/16,1) => (batch,h,w,1) #out 3
        
        # reverse attention branch 2
        resized_s3 = self.resize_2(s3)# resize (batch, h/16,w/16,1) => (batch, h/8,w/8,1)
        s2 = self.ra_2(self.features[1],resized_s3)
        lateral_out_s2 = self.resize_s2(s2)# resize (batch,h/8,w/8,1) => (batch,h,w,1) #out 2

        return lateral_out_sg, lateral_out_s4, lateral_out_s3, lateral_out_s2

    def compile(
        self, 
        optimizer: tf.keras.optimizers.Optimizer, 
        loss: tf.keras.losses.Loss, 
        train_metric: tf.keras.metrics.Metric,
        val_metric: tf.keras.metrics.Metric,
        loss_weights: list = [1,1,1,1],
        **kwargs
    ):
        super(PRAresnet, self).compile(**kwargs)
        assert len(loss_weights) == 4
        self.optim = optimizer
        self.loss_fn = loss
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.loss_weights = loss_weights

    @tf.function
    def train_step(self, x_img: tf.Tensor, y_mask: tf.Tensor, gclip:float):
        with tf.GradientTape() as tape:
            lateral_out_sg, lateral_out_s4, lateral_out_s3, lateral_out_s2 = self(x_img, training=True)
            loss1 = self.loss_fn(y_mask, lateral_out_sg)
            loss2 = self.loss_fn(y_mask, lateral_out_s4)
            loss3 = self.loss_fn(y_mask, lateral_out_s3)
            loss4 = self.loss_fn(y_mask, lateral_out_s2)

            train_loss = (self.loss_weights[0]*loss1) + (self.loss_weights[1]*loss2) + \
                            (self.loss_weights[2]*loss3) + (self.loss_weights[-1]*loss4)

        # get gradients, clip gradient with some margin and apply to optimizer
        grads = tape.gradient(train_loss, self.trainable_variables)
        grads = [(tf.clip_by_value(grad, clip_value_min=-gclip, clip_value_max=gclip))
                for grad in grads]

        self.optim.apply_gradients(zip(grads, self.trainable_variables))

        # update metrics
        self.train_metric.update_state(y_mask, lateral_out_s2)

        return train_loss
    
    @tf.function
    def test_step(self, x_img: tf.Tensor, y_mask: tf.Tensor):
        lateral_out_sg, lateral_out_s4, lateral_out_s3, lateral_out_s2 = self(x_img, training=False)
        loss1 = self.loss_fn(y_mask, lateral_out_sg)
        loss2 = self.loss_fn(y_mask, lateral_out_s4)
        loss3 = self.loss_fn(y_mask, lateral_out_s3)
        loss4 = self.loss_fn(y_mask, lateral_out_s2)

        val_loss = (self.loss_weights[0]*loss1) + (self.loss_weights[1]*loss2) + \
                            (self.loss_weights[2]*loss3) + (self.loss_weights[-1]*loss4)

        self.val_metric.update_state(y_mask, lateral_out_s2)

        return val_loss
    


    def build_graph(self, inshape:tuple) -> tf.keras.Model:
        '''
        Custom method just to see graph summary of model
        reference: https://github.com/tensorflow/tensorflow/issues/31647#issuecomment-692586409
        '''
        x = tf.keras.layers.Input(shape=inshape)
        return tf.keras.Model(inputs=[x], outputs = self.call(x), name='PRAnet')


# test model class
if __name__ == "__main__":
    from time import time
    tf.random.set_seed(3)
    raw_inputs = (352,352,3)
    pranet = PRAresnet(IMG_H=352,IMG_W=352,filters=32)
    start_time = time()
    out = pranet(tf.random.normal(shape=(8, *raw_inputs)))
    end_time = time()
    print(pranet.build_graph(inshape=(raw_inputs)).summary())
    for o in out:
        print(o.shape)
    print(f"time taken for single forward pass: {(end_time-start_time)}")