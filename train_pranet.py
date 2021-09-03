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

from datetime import datetime
from time import time
from tqdm import tqdm
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.losses import WBCEDICELoss
from utils.dataset import TfdataPipeline
from model.PRA_net import PRAnet
import tensorflow as tf
tf.random.set_seed(41)


def process_output(x: tf.Tensor, threshold:float = None):
    '''
    Post processing feature and output tensor that will be logged for Tensorboard
    '''

    x = tf.sigmoid(x)
    if threshold:
        x = tf.cast(tf.math.greater(x, threshold), tf.float32)
    x = x * 255.0
    return x


def train(
    dataset_dir: str,
    trained_model_dir: str,
    img_size: int = 352,
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    gclip: float = 1.0,
    dataset_split: float = 0.1,
    backbone_trainable: bool = True,
    backbone_arc:str = 'resnet50', 
    logdir: str = "logs/",
):
    assert os.path.isdir(dataset_dir)
    if backbone_arc == 'mobilenetv2' and img_size > 224:
        tf.print(f"For backbone {backbone_arc} inputsize should be 32 < inputsize <=224")
        sys.exit()
    
    if not os.path.exists(dataset_dir):
        print(f"No dir named {dataset_dir} exist")
        sys.exit()

    if not os.path.exists(trained_model_dir):
        os.mkdir(path=trained_model_dir)

    # instantiate tf.summary writer
    logsdir = logdir + "PRAnet/" + "PRAnet_"+backbone_arc+datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(logsdir + "/train/")
    val_writer = tf.summary.create_file_writer(logsdir + "/val/")

    # initialize tf.data pipeline
    tf_datapipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_dir,
        IMG_H=img_size,
        IMG_W=img_size,
        batch_size=batch_size,
        split=dataset_split
    )
    train_data = tf_datapipeline.data_loader(dataset_type='train')
    val_data = tf_datapipeline.data_loader(dataset_type='valid')

    # instantiate optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
    )

    # instantiate loss function
    loss_fn = WBCEDICELoss(name='w_bce_dice_loss')


    # instantiate model (PRAnet)
    pranet = PRAnet(
        IMG_H=img_size,
        IMG_W=img_size,
        filters=32,
        backbone_arch=backbone_arc,
        backbone_trainable=backbone_trainable
    )

    # compile the model
    pranet.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )
    tf.print(pranet.build_graph(inshape=(img_size, img_size, 3)).summary())
    tf.print("==========Model configs==========")
    tf.print(
        f"Training and validating PRAnet for {epochs} epochs \nlearing_rate: {lr} \nInput shape:({img_size},{img_size},3) \nBatch size: {batch_size} \nBackbone arc: {backbone_arc} \nBackbone Trainable: {backbone_trainable}"
    )
    # train for epochs
    for e in range(epochs):
        t = time()

        for (x_train_img, y_train_mask) in tqdm(train_data, unit='steps', desc='training...', colour='red'):
            train_loss, train_dice, train_iou = pranet.train_step(
                x_img=x_train_img, y_mask=y_train_mask, gclip=gclip)

        for (x_val_img, y_val_mask) in tqdm(val_data, unit='steps', desc='Validating...', colour='green'):
            val_loss, val_dice, val_iou = pranet.test_step(x_img=x_val_img, y_mask=y_val_mask)

        tf.print(
            "ETA:{} - epoch: {} - loss: {} - dice: {} - IoU: {} - val_loss: {} - val_dice: {} - val_IoU: {} \n".format(
                round((time() - t)/60, 2), (e+1), train_loss, train_dice, train_iou, val_loss, val_dice, val_iou)
            )
        

        tf.print("Writing to Tensorboard...")
        lateral_out_sg, lateral_out_s4, lateral_out_s3, lateral_out_s2 = pranet(x_val_img, training=False)
        lateral_out_sg = process_output(lateral_out_sg)
        lateral_out_s4 = process_output(lateral_out_s4)
        lateral_out_s3 = process_output(lateral_out_s3)
        lateral_out_s2 = process_output(lateral_out_s2, threshold = 0.3)


        with train_writer.as_default():
            tf.summary.scalar(name='train_loss', data=train_loss, step=e+1)
            tf.summary.scalar(name='dice', data = train_dice, step=e+1)
            tf.summary.scalar(name='iou', data = train_iou, step=e+1)

        
        with val_writer.as_default():
            tf.summary.scalar(name='val_loss', data=val_loss, step=e+1)
            tf.summary.scalar(name='val_dice', data=val_dice, step=e+1)
            tf.summary.scalar(name='val_dice', data=val_iou, step=e+1)
            tf.summary.image(name='Y_mask', data=y_val_mask*255, step=e+1, max_outputs=batch_size, description='Val data')
            tf.summary.image(name='Global S Map', data=lateral_out_sg, step=e+1, max_outputs=batch_size, description='Val data')
            tf.summary.image(name='S4 Map', data=lateral_out_s4, step=e+1, max_outputs=batch_size, description='Val data')
            tf.summary.image(name='S3 Map', data=lateral_out_s3, step=e+1, max_outputs=batch_size, description='Val data')
            tf.summary.image(name='S2 Map', data=lateral_out_s2, step=e+1, max_outputs=batch_size, description='Val data')
        
        if (e+1)%5 == 0:
            tf.print(
                f"Saving model at {trained_model_dir}..."
            )
            pranet.save(f"{trained_model_dir}pranet_{backbone_arc}", save_format='tf')
            tf.print(f"model saved at {trained_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='polyps_dataset/', help='path to dataset')
    parser.add_argument('--data_split', type=float,
                        default=0.1, help='split percent of val and test data')
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--inputsize', type=int,
                        default=352, help='input image size')
    parser.add_argument('--backbone', type=str,
                        default='resnet50', help='Feature Extractor backbone Arc')
    parser.add_argument('--gclip', type=float,
                        default=1.0, help='gradient clipping margin')
    parser.add_argument('--trained_model_path', type=str,
                        default='trained_model/')
    parser.add_argument('--logdir', type=str, help="Tensorboard logs",
                        default='logs/')

    opt = parser.parse_args()

    train(
        dataset_dir=opt.data_path,
        img_size=opt.inputsize,
        batch_size=opt.batchsize,
        epochs=opt.epoch,
        lr=opt.lr,
        gclip=opt.gclip,
        trained_model_dir=opt.trained_model_path,
        dataset_split=opt.data_split,
        backbone_trainable=True,
        backbone_arc = opt.backbone,
        logdir=opt.logdir
    )
