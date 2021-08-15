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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_mask(path: str) -> tf.Tensor:
    mask_raw = tf.io.read_file(path)
    mask = tf.io.decode_jpeg(mask_raw, channels=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.image.resize(mask,[512, 512])
    mask = mask/255.0
    mask = tf.expand_dims(mask, axis=0)

    return mask


def read_image(path: str) -> tf.Tensor:
    image_raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_raw, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, [512,512])
    image = image/255.0
    image = tf.expand_dims(image, axis=0)
    return image


def get_weights(mask: tf.Tensor, ksize: int = 31):
    mask_weights = 1 + 10 * \
        tf.abs(tf.nn.avg_pool2d(mask, ksize=31, strides=1, padding="SAME")-mask)

    # normalizing just for visualizing purposes
    mask_weights = mask_weights / tf.reduce_max(mask_weights)
    return mask_weights


def vis_iou_bce_weights(image,weight_map):
    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1,3, width_ratios = [5,5,5])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(grid_spec[1])
    plt.imshow(weight_map)
    plt.axis('off')
    plt.title("BCE/IOU loss Weights")
    
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(weight_map, alpha=0.6)
    plt.axis('off')
    plt.title("Image + Alpha Map")

    plt.grid('off')
    plt.show()
    



if __name__ == "__main__":
    path_to_mask = "polyps_dataset/masks/cjxn4fm0wg1cn0738rvy81d2v.jpg"
    path_to_image = "polyps_dataset/images/cjxn4fm0wg1cn0738rvy81d2v.jpg"

    image = read_image(path_to_image)
    image = tf.squeeze(image)

    mask = read_mask(path=path_to_mask)
    mask_weight = get_weights(mask=mask)
    mask_weight = tf.squeeze(mask_weight)
    mask = tf.squeeze(mask)

    vis_iou_bce_weights(image,mask_weight)
