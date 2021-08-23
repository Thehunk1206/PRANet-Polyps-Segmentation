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
from model.conv_module import ConvModule


class PartialDecoder(tf.keras.layers.Layer):
    def __init__(self, filters: int, name:str):
        super(PartialDecoder, self).__init__(name=name)
        self.filters = filters

        self.upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.conv_up1 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up2 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up3 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up4 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up5 = ConvModule(filters=2*filters, kernel_size=(3, 3))

        self.conv_concat_1 = ConvModule(filters=2*filters, kernel_size=(3, 3))
        self.conv_concat_2 = ConvModule(filters=3*filters, kernel_size=(3, 3))

        self.conv4 = ConvModule(filters=3*filters, kernel_size=(3, 3))
        self.conv5 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, rfb_feat1: tf.Tensor, rfb_feat2: tf.Tensor, rfb_feat3: tf.Tensor) -> tf.Tensor:
        rfb_feat1 = tf.nn.relu(rfb_feat1)
        rfb_feat2 = tf.nn.relu(rfb_feat2)
        rfb_feat3 = tf.nn.relu(rfb_feat3)

        x1_1 = rfb_feat1
        x2_1 = self.conv_up1(self.upsampling(rfb_feat1)) * rfb_feat2
        x3_1 = self.conv_up2(self.upsampling(self.upsampling(rfb_feat1)))  \
            * self.conv_up3(self.upsampling(rfb_feat2)) * rfb_feat3

        x2_2 = tf.concat([x2_1, self.conv_up4(
            self.upsampling(x1_1))], axis=-1)
        x2_2 = self.conv_concat_1(x2_2)

        x3_2 = tf.concat([x3_1, self.conv_up5(self.upsampling(x2_2))], axis=-1)
        x3_2 = self.conv_concat_2(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


# test the module
if __name__ == "__main__":
    ppd = PartialDecoder(32, name="partial_decoder")
    # first call to the `ppd` will create weights
    feat3 = tf.ones(shape=(8, 44, 44, 32))
    feat2 = tf.ones(shape=(8, 22, 22, 32))
    feat1 = tf.ones(shape=(8, 11, 11, 32))
    y = ppd(feat1, feat2, feat3)
    print("weights:", len(ppd.weights))
    print("trainable weights:", len(ppd.trainable_weights))
    print("config:", ppd.get_config())
    print(f"Y: {y.shape}")
