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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.python import keras

from model.backbone import FE_backbone

sample_image_path = 'polyps_dataset/images/cju1ddr6p4k5z08780uuuzit2.jpg'
image_size = 1024


def preprocess_input_image(path: str, image_size: int) -> tf.Tensor:
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    # Add another dim of bactch size (batch,h,w,c)
    image = tf.expand_dims(image, axis=0)
    image = resnet.preprocess_input(image)

    return image


def get_features_of_image(path: str, model: keras.Model) -> list:
    image = preprocess_input_image(path=path, image_size=image_size)
    extracted_feature = model(image)
    return extracted_feature


if __name__ == "__main__":

    model = FE_backbone(inshape=(image_size, image_size, 3)).get_fe_backbone()
    extracted_feature = get_features_of_image(
        path=sample_image_path, model=model)

    # https://www.analyticsvidhya.com/blog/2020/11/tutorial-how-to-visualize-feature-maps-directly-from-cnn-layers/ refer to this article
    for feat in extracted_feature[0]:
        feat = tf.expand_dims(feat, axis=0)
        k = feat.shape[-1]
        size = feat.shape[1]
        image_belt = np.zeros((size, size*10), dtype=np.int)
        for i in range(10):
            feature_image = feat[0, :, :, i+30]
            feature_image -= tf.reduce_mean(feature_image)
            feature_image /= tf.math.reduce_std(feature_image)
            feature_image *= 64
            feature_image += 128
            feature_image = tf.cast(tf.clip_by_value(
                feature_image, 0, 255), dtype=tf.int8)
            image_belt[:, i * size: (i + 1) * size] = feature_image
        scale = 20. / 10
        plt.figure(figsize=(scale * k, scale))
        plt.grid(False)
        plt.imshow(image_belt, aspect='auto')
        plt.show()
