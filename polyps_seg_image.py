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


def process_output(x: tf.Tensor, original_img: tf.Tensor, threshold: float = None) -> tf.Tensor:
    x = tf.sigmoid(x)

    if threshold:
        x = tf.cast(tf.math.greater(x, threshold), tf.float32)
    
    x = tf.squeeze(x, axis=0)
    x = tf.image.resize(x, [original_img.shape[0], original_img.shape[1]], ResizeMethod.BICUBIC)
    # we use tf.tile to make multiple copy of output single channel image
    mutiple_const = tf.constant([1,1,3]) # [1,1,3] h(1)xw(1)xc(3)
    x = tf.tile(x,mutiple_const)
    
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


def vis_predicted_mask(*images: tf.Tensor):
    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(2, 3, width_ratios=[3, 3, 3])

    plt.subplot(grid_spec[0])
    plt.imshow(images[0])
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(grid_spec[1])
    plt.imshow(images[1])
    plt.axis('off')
    plt.title("Predicted mask")

    plt.subplot(grid_spec[2])
    plt.imshow(images[0])
    plt.imshow(images[1], alpha=0.6)
    plt.axis('off')
    plt.title("Image +  Predicted Mask")

    plt.subplot(grid_spec[3])
    plt.imshow(images[2])
    plt.axis('off')
    plt.title("Global S Map")

    plt.subplot(grid_spec[4])
    plt.imshow(images[3])
    plt.axis('off')
    plt.title("Side Map 4")

    plt.subplot(grid_spec[5])
    plt.imshow(images[4])
    plt.axis('off')
    plt.title("Side Map 3")

    plt.grid('off')
    plt.savefig(f"detection_{time()}.jpg")


def run(
    model_path: str,
    image_path: str,
    imgsize: int = 352,
    threshold: float = 0.5
):
    assert os.path.exists(model_path)
    assert os.path.exists(image_path)
    assert 1.0 > threshold > 0.0

    pranet = get_model(model_path=model_path)

    input_image, original_image = read_image(
        path=image_path, img_size=imgsize)

    tf.print("[info] Computing output mask..")
    start = time()
    outs = pranet(input_image)
    end = time()
    sg, s4, s3, final_out = outs
    final_out = process_output(final_out, original_img=original_image, threshold=threshold)
    sg = process_output(sg, original_img=original_image)
    s4 = process_output(s4, original_img=original_image)
    s3 = process_output(s3, original_img=original_image)


    total_time = round((end - start)*1000, ndigits=2)
    tf.print(f"Total runtime of model: {total_time}ms")


    vis_predicted_mask(original_image, final_out, sg, s4, s3)

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
