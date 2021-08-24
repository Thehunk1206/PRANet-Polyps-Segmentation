import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from tqdm.notebook import tqdm
from time import time

from tensorflow.python.data.ops.dataset_ops import DatasetV2
from utils.losses_and_metrics import dice_coef
from utils.dataset import TfdataPipeline
from tensorflow.keras import models
import tensorflow as tf



def get_model(model_path: str):
    assert isinstance(model_path, str)

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model("trained_model/pranet_v1.1")

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
    test_data = datapipeline(dataset_path=dataset_path, imgsize=352)

    dice_coefs = []
    runtimes = []

    for (image, mask) in tqdm(test_data):
        start = time()
        outs = pranet(image)
        end = time()
        final_out = tf.sigmoid(outs[-1])
        final_out = tf.cast(tf.math.greater(final_out, 0.5), tf.float32)

        total_time = round((end - start)*1000, ndigits=3)
        dice = dice_coef(y_mask=mask, y_pred=final_out)
        dice_coefs.append(dice)
        runtimes.append(total_time)
    
    mean_dice = sum(dice_coefs)/len(dice_coefs)
    mean_runtime = sum(runtimes)/ len(runtimes)
    tf.print(
            f"Average runtime of model: {mean_runtime}\n",
            f"Mean Dice coef: {mean_dice}\n\n\n",
            "NOTE: The runtime of model can be high at first run as it \ntake time cache the data in memory.\ntry to run the script again without closing the session"
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
    
