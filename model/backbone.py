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
import sys

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


class FE_backbone():
    def __init__(self, model_architecture: str = 'resnet50', inshape: tuple = (352, 352, 3), is_trainable: bool = True):
        self.inshape = inshape
        self._supported_arch = ['resnet50', 'mobilenetv2']
        self.model_architecture = model_architecture
        if self.model_architecture not in self._supported_arch:
            tf.print(
                f"Model Architecture should be one of {self._supported_arch}")
            sys.exit()

        self.is_trainable = is_trainable
        self.resnet_feature_extractor_layer_name = [
            'conv2_block3_out',  # 256
            'conv3_block4_out',  # 512
            'conv4_block6_out',  # 1024
            'conv5_block3_out',  # 2048
        ]
        self.mobilenet_feature_extractor_layer_name = [
            'block_3_expand_relu',  # 56x56x144
            'block_6_expand_relu',  # 28x28x192
            'block_13_expand_relu',  # 14x14x576
            'out_relu',             # 7x7x1280
        ]

        if self.model_architecture == 'resnet50':
            self.backbone = ResNet50(
                include_top=False, input_shape=self.inshape
            )

        if self.model_architecture == 'mobilenetv2':
            self.backbone = MobileNetV2(
                include_top=False, input_shape=self.inshape
            )

        self.backbone.trainable = self.is_trainable

    def get_fe_backbone(self) -> tf.keras.Model:

        layer_out = []
        if self.model_architecture == 'resnet50':
            for layer_name in self.resnet_feature_extractor_layer_name:
                layer_out.append(self.backbone.get_layer(layer_name).output)

        if self.model_architecture == 'mobilenetv2':
            for layer_name in self.mobilenet_feature_extractor_layer_name:
                layer_out.append(self.backbone.get_layer(layer_name).output)

        fe_backbone_model = tf.keras.models.Model(
            inputs=self.backbone.input, outputs=layer_out, name='resnet50_include_top_false' if self.model_architecture == 'resnet50' else 'mobilenetv2_include_top_false')

        return fe_backbone_model


# test the module
if __name__ == "__main__":
    fe_backone = FE_backbone(model_architecture='resnet50', inshape=(224,224,3))
    model = fe_backone.get_fe_backbone()
    print(model.output)
    image_raw = tf.io.read_file(
        'polyps_dataset/images/cju0qkwl35piu0993l0dewei2.jpg')
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, dtype=tf.float32)
    image = image/255.0
    image = tf.expand_dims(image, axis=0)
    side_features = model(image)
    print(model.summary())
    side_features1, side_features2, side_features3 = side_features[
        0], side_features[1], side_features[2]
