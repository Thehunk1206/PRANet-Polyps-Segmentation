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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from tqdm import tqdm
from time import time

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from utils.losses_and_metrics import dice_coef, iou_metric
from utils.dataset import TfdataPipeline


def get_model(model_path: str):
    assert isinstance(model_path, str)

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model(model_path)

    tf.print(
        "loaded model {}".format(model)
    )
    return model


def datapipeline(dataset_path: str, imgsize:int = 352) -> DatasetV2:
    assert isinstance(dataset_path, str)

    tfpipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_path, IMG_H=imgsize, IMG_W=imgsize, batch_size=1)
    test_data = tfpipeline.data_loader(dataset_type='test')

    return test_data


def run_test(
    model_path:str,
    imgsize:int = 352,
    dataset_path:str='polyps_dataset/'
):
    pranet = get_model(model_path=model_path)
    test_data = datapipeline(dataset_path=dataset_path, imgsize=imgsize)

    dice_coefs = []
    ious = []
    runtimes = []

    for (image, mask) in tqdm(test_data, desc='Testing..', unit='steps', colour='green'):
        start = time()
        outs = pranet(image)
        end = time()
        final_out = tf.sigmoid(outs[-1])
        final_out = tf.cast(tf.math.greater(final_out, 0.5), tf.float32)

        total_time = round((end - start)*1000, ndigits=2)
        dice = dice_coef(y_mask=mask, y_pred=final_out)
        iou = iou_metric(y_mask=mask, y_pred=final_out)
        dice_coefs.append(dice)
        ious.append(iou)
        runtimes.append(total_time)

    mean_dice = sum(dice_coefs)/len(dice_coefs)
    mean_iou = sum(ious)/len(ious)
    mean_runtime = sum(runtimes[3:])/ len(runtimes[3:])
    tf.print(
            f"Average runtime of model: {mean_runtime}ms\n",
            f"Mean IoU: {mean_iou}\n"
            f"Mean Dice coef: {mean_dice}\n\n\n",
            "NOTE: The runtime of model can be high at first run as it \ntake time to cache the data in memory.\ntry to run the script again without closing the session"
        )


if __name__ == "__main__":
    __description__ = '''
    Python script to test the model's performance on test data
    '''
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--model_path', type=str,
                        help='path to savedModel', required=True)
    parser.add_argument('--data_path', type=str,
                        default='polyps_dataset/', help='path to dataset')
    parser.add_argument('--inputsize', type=int,
                        default=352, help='input image size')

    opt = parser.parse_args()

    run_test(
        model_path=opt.model_path,
        imgsize=opt.inputsize,
        dataset_path=opt.data_path
    )
    
