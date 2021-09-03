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

from utils.dataset import TfdataPipeline
from utils.segmentation_metric import dice_coef, iou_metric, MAE, WFbetaMetric, SMeasure, Emeasure
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.keras import models
import tensorflow as tf
from time import time
from tqdm import tqdm
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def datapipeline(dataset_path: str, imgsize: int = 352) -> DatasetV2:
    assert isinstance(dataset_path, str)

    tfpipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_path, IMG_H=imgsize, IMG_W=imgsize, batch_size=1)
    test_data = tfpipeline.data_loader(dataset_type='test')

    return test_data


def run_test(
    model_path: str,
    imgsize: int = 352,
    dataset_path: str = 'polyps_dataset/',
    threshold: float = 0.5
):
    assert os.path.exists(model_path)
    assert os.path.exists(dataset_path)
    assert 1.0 > threshold > 0.0

    pranet = get_model(model_path=model_path)
    test_data = datapipeline(dataset_path=dataset_path, imgsize=imgsize)

    # initialize metrics
    wfb_metric = WFbetaMetric()
    smeasure_metric = SMeasure()
    emeasure_metric = Emeasure()
    # collect metric for individual test data to average it later
    dice_coefs = []
    ious = []
    wfbs = []
    smeasures = []
    emeasures = []
    maes = []
    runtimes = []

    for (image, mask) in tqdm(test_data, desc='Testing..', unit='steps', colour='green'):
        start = time()
        outs = pranet(image)
        end = time()
        # squesh the out put between 0-1
        final_out = tf.sigmoid(outs[-1])
        # convert the out map to binary map
        final_out = tf.cast(tf.math.greater(final_out, 0.5), tf.float32)

        total_time = round((end - start)*1000, ndigits=2)

        dice = dice_coef(y_mask=mask, y_pred=final_out)
        iou = iou_metric(y_mask=mask, y_pred=final_out)
        mae = MAE(y_mask=mask, y_pred= final_out)
        wfb = wfb_metric(y_mask=mask, y_pred=final_out)
        smeasure = smeasure_metric(y_mask=mask, y_pred=final_out)
        emeasure = emeasure_metric(y_mask=mask, y_pred= final_out)

        dice_coefs.append(dice)
        ious.append(iou)
        maes.append(mae)
        wfbs.append(wfb)
        smeasures.append(smeasure)
        emeasures.append(emeasure)
        runtimes.append(total_time)

    mean_dice = sum(dice_coefs)/len(dice_coefs)
    mean_iou = sum(ious)/len(ious)
    mean_mae = sum(maes)/len(maes)
    mean_wfb = sum(wfbs)/len(wfbs)
    mean_smeasure = sum(smeasures)/len(smeasures)
    mean_emeasure = sum(emeasures)/len(emeasures)
    mean_runtime = sum(runtimes[3:]) / len(runtimes[3:])
    tf.print(
        f"Average runtime of model: {mean_runtime}ms \n",
        f"Mean IoU: {mean_iou}\n",
        f"Mean Dice coef: {mean_dice}\n",
        f"Mean wfb: {mean_wfb}\n",
        f"Mean Smeasure: {mean_smeasure}\n",
        f"Mean Emeasure: {mean_emeasure}\n",
        f"MAE: {mean_mae}\n",
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
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='setting the threshold to convert out map to binary')

    opt = parser.parse_args()

    run_test(
        model_path=opt.model_path,
        imgsize=opt.inputsize,
        dataset_path=opt.data_path,
        threshold=opt.threshold
    )
