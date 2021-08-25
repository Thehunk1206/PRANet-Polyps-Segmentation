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
import argparse
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def read_image(path: str, img_size: int = 352) -> tf.Tensor:
    image_raw = tf.io.read_file(path)
    original_image = tf.io.decode_jpeg(image_raw, channels=3)
    original_image = tf.cast(original_image, dtype=tf.float32)
    original_image = original_image/255.0

    resized_image = tf.image.resize(original_image, [img_size, img_size])
    resized_image = tf.expand_dims(resized_image, axis=0)

    return resized_image, original_image


def process_output(x: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    x = tf.sigmoid(x)
    x = tf.cast(tf.math.greater(x, threshold), tf.float32)
    return x


def get_model(model_path: str):
    assert isinstance(model_path, str)

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model(model_path)

    tf.print(
        "[info] loaded model"
    )
    return model


def vis_predicted_mask(image: tf.Tensor, pred: tf.Tensor):
    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 5])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(grid_spec[1])
    plt.imshow(pred)
    plt.axis('off')
    plt.title("Predicted mask")

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(pred, alpha=0.6)
    plt.axis('off')
    plt.title("Image +  Predicted Mask")

    plt.grid('off')
    plt.savefig(f"detection {time()}.jpg")


def run(
    model_path: str,
    image_path: str,
    imgsize: int = 352,
):
    assert os.path.exists(model_path)
    assert os.path.exists(image_path)

    pranet = get_model(model_path=model_path)

    input_image, original_image = read_image(
        path=image_path, img_size=imgsize)

    tf.print("[info] Computing output mask..")
    start = time()
    outs = pranet(input_image)
    end = time()
    final_out = outs[-1]
    final_out = process_output(final_out, threshold=0.5)

    total_time = round((end - start)*1000, ndigits=2)
    tf.print(f"Total runtime of model: {total_time}ms")

    final_out = tf.squeeze(final_out, axis=0)
    final_out = tf.image.resize(final_out, [original_image.shape[0], original_image.shape[1]], ResizeMethod.BICUBIC)
    # we use tf.tile to make multiple copy of output single channel image
    mutiple_const = tf.constant([1,1,3]) # [1,1,3] h(1)xw(1)xc(3)
    final_out = tf.tile(final_out,mutiple_const)

    vis_predicted_mask(image=original_image, pred=final_out)

if __name__ == "__main__":
    __description__ = '''
    Python script to compute mask map for single Image
    '''
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--model_path', type=str,
                        help='path to savedModel', required=True)
    parser.add_argument('--image_path', type=str, 
                        help='path to dataset', required=True)
    parser.add_argument('--inputsize', type=int,
                        default=352, help='input image size')

    opt = parser.parse_args()

    run(
        model_path=opt.model_path,
        image_path=opt.image_path,
        imgsize=opt.inputsize
    )
