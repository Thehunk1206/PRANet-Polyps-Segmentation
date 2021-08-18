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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
from tqdm import tqdm
from time import time

import tensorflow as tf
from model.PRA_resenet import PRAresnet
from utils.dataset import TfdataPipeline
from utils.losses_and_metrics import WBCEIOULoss, DiceCoef


def train(
    dataset_dir: str,
    trained_model_dir: str,
    img_size: int = 352,
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    gclip: float = 1.0
):
    assert os.path.isdir(dataset_dir)

    if not os.path.exists(dataset_dir):
        print(f"No dir named {dataset_dir} exist")
        sys.exit()

    if not os.path.exists(trained_model_dir):
        os.mkdir(path=trained_model_dir)

    # initialize tf.data pipeline
    tf_datapipeline = TfdataPipeline(
        BASE_DATASET_DIR="polyps_dataset/",
        IMG_H=img_size,
        IMG_W=img_size,
        batch_size=batch_size,
        split=0.3
    )
    train_data = tf_datapipeline.data_loader(dataset_type='train')
    val_data = tf_datapipeline.data_loader(dataset_type='valid')

    # instantiate optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
    )

    # instantiate loss function
    loss_fn = WBCEIOULoss(name='w_bce_iou_loss')

    # instantiate metric function
    train_metric = DiceCoef(name='train_dice_coeff_metric')
    val_metric = DiceCoef(name='val_dice_coeff_mettric')

    # instantiate model (PRAresnet)
    praresnet = PRAresnet(
        IMG_H=img_size,
        IMG_W=img_size,
        filters=32
    )

    # compile the model
    praresnet.compile(
        optimizer=optimizer,
        loss=loss_fn,
        train_metric=train_metric,
        val_metric=val_metric
    )
    tf.print(praresnet.build_graph(inshape=(img_size,img_size,3)).summary())
    tf.print("==========Model configs==========")
    tf.print(
        f"Training and validating PRAresnet for {epochs} epochs \nlearing_rate: {lr} \nInput shape:({img_size},{img_size},3) \nBatch size: {batch_size}"
    )
    for e in range(epochs):
        t = time()

        for (x_train_img, y_train_mask) in tqdm(train_data, unit='steps', desc='training...', colour='red'):
            train_loss = praresnet.train_step(
                x_img=x_train_img, y_mask=y_train_mask, gclip=gclip)

        for (x_val_img, y_val_mask) in tqdm(val_data, unit='steps', desc='Validating...', colour='green'):
            val_loss = praresnet.test_step(x_img=x_val_img, y_mask=y_val_mask)

        tf.print(
            "ETA:{} - epoch: {} - loss: {} - dice: {} - val_loss: {} - val_dice: {}\n".format(
                round((time() - t)/60, 2), (e+1), train_loss, float(
                    train_metric.result()), val_loss, float(val_metric.result())
            )
        )
        train_metric.reset_states()
        val_metric.reset_states()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='polyps_dataset/', help='path to dataset')
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--inputsize', type=int,
                        default=352, help='input image size')
    parser.add_argument('--gclip', type=float,
                        default=1.0, help='gradient clipping margin')
    parser.add_argument('--trained_model_path', type=str,
                        default='trained_model/')

    opt = parser.parse_args()

    train(
        dataset_dir=opt.data_path,
        img_size= opt.inputsize,
        batch_size= opt.batchsize,
        epochs=opt.epoch,
        lr=opt.lr,
        gclip=opt.gclip,
        trained_model_dir=opt.trained_model_path
    )

