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
import os

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python import keras


class FE_backbone():
    def __init__(self, inshape: tuple = (352, 352, 3), is_trainable: bool = True):
        super(FE_backbone, self).__init__()
        self.inshape = inshape
        self.is_trainable = is_trainable
        self.resnet_backbone = ResNet50(
            include_top=False, input_shape=self.inshape
        )
        self.resnet_backbone.trainable = self.is_trainable

        self.feature_extractor_layer_name = [
            'conv2_block3_out',  # 256
            'conv3_block4_out',  # 512
            'conv4_block6_out',  # 1024
            'conv5_block3_out',  # 2048
        ]

    def get_fe_backbone(self):

        layer_out = []
        for layer_name in self.feature_extractor_layer_name:
            layer_out.append(self.resnet_backbone.get_layer(layer_name).output)

        fe_backbone_model = keras.models.Model(
            inputs=self.resnet_backbone.input, outputs=layer_out)

        return fe_backbone_model


# if __name__ == "__main__":
    # fe_resnet = FE_backbone()
    # model = fe_resnet.get_fe_backbone()
    # print(model.output)
    # image_raw = tf.io.read_file('polyps_dataset/images/cju0qkwl35piu0993l0dewei2.jpg')
    # image = tf.image.decode_jpeg(image_raw,channels=3)
    # image = tf.image.resize(image,[352,352])
    # image = tf.cast(image, dtype=tf.float32)
    # image = image/255.0
    # image = tf.expand_dims(image, axis=0)
    # side_features = model(image)
    # side_features1,side_features2,side_features3 = side_features[0], side_features[1],side_features[2]
