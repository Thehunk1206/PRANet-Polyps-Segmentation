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


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self, filters: int,
        kernel_size: tuple,
        strides: tuple = (1, 1),
        padding: str = "same",
        dilation_rate: tuple = (1, 1),
    ):
        super(ConvModule, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)

        return x

    def get_config(self) -> dict:
        config = super(ConvModule, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


# test the module
if __name__ == "__main__":
    cm = ConvModule(96, (1, 3), (1, 1))
    # first call to the `cm` will create weights
    y = cm(tf.ones(shape=(2, 32, 32, 512)))

    print("weights:", len(cm.weights))
    print("trainable weights:", len(cm.trainable_weights))
    print("config:", cm.get_config())
    print(f"Y: {y.shape}")
