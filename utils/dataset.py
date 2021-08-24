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
import sys
import glob

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2


class TfdataPipeline:
    def __init__(
        self,
        BASE_DATASET_DIR: str,
        IMG_H: int = 512,
        IMG_W: int = 512,
        IMG_C: int = 3,
        batch_size: int = 32,
        split: float = 0.1
    ) -> None:
        self.BASE_DATASET_DIR = BASE_DATASET_DIR
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C
        self.batch_size = batch_size
        self.split = split
        self.__datasettype = ['train', 'valid', 'test']

        if not os.path.exists(BASE_DATASET_DIR):
            print(
                f"[Error] Dataset directory {BASE_DATASET_DIR} does not exist!")
            sys.exit()

    def __load_and_split_dataset_files(self, path: str, split: float):
        '''Loads the path name of each images and masks
        and split it in ratio 80:10:10(train:valid:test)'''

        images = sorted(glob.glob(os.path.join(path, 'images/*')))
        mask = sorted(glob.glob(os.path.join(path, 'masks/*')))
        total_items = len(images)
        valid_items = int(total_items * split)
        test_items = int(total_items * split)

        train_x, valid_x = train_test_split(
            images, test_size=valid_items, random_state=12)
        train_y, valid_y = train_test_split(
            mask, test_size=valid_items, random_state=12)

        train_x, test_x = train_test_split(
            train_x, test_size=test_items, random_state=12)
        train_y, test_y = train_test_split(
            train_y, test_size=test_items, random_state=12)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    def __read_image_and_mask(self, image_path: str, mask_path: str) -> tf.Tensor:

        img_raw = tf.io.read_file(image_path)
        mask_raw = tf.io.read_file(mask_path)

        img = tf.io.decode_jpeg(img_raw)
        mask = tf.io.decode_jpeg(mask_raw, channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32) # normalize the values between 0-1

        mask = tf.image.convert_image_dtype(mask, tf.float32) # normalize the values between 0-1

        img = tf.image.resize(img, [self.IMG_H, self.IMG_W])
        mask = tf.image.resize(mask, [self.IMG_H, self.IMG_W])

        return img, mask

    def __tf_dataset(self, images_path: str, mask_path: str) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((images_path, mask_path))
        # dataset = dataset.shuffle(buffer_size=self.batch_size*2)
        dataset = dataset.map(
            self.__read_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE).cache()
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def data_loader(self, dataset_type: str = 'train') -> DatasetV2:
        '''
        dataset_type should be in ['train','valid','test']
        '''
        if dataset_type not in self.__datasettype:
            print(
                f"[Error] invalid option {dataset_type} option should be in {self.__datasettype}")
            sys.exit()
        (train_x, train_y), (valid_x, valid_y), (test_x,
                                                 test_y) = self.__load_and_split_dataset_files(self.BASE_DATASET_DIR, split=self.split)

        if dataset_type == 'train':
            train_dataset = self.__tf_dataset(train_x, train_y)
            return train_dataset
        elif dataset_type == 'valid':
            valid_dataset = self.__tf_dataset(valid_x, valid_y)
            return valid_dataset
        elif dataset_type == 'test':
            test_dataset = self.__tf_dataset(test_x, test_y)
            return test_dataset


if __name__ == "__main__":
    from tqdm import tqdm
    tf_datapipeline = TfdataPipeline(BASE_DATASET_DIR="polyps_dataset/")
    train_data = tf_datapipeline.data_loader()

    for image, mask in tqdm(train_data, unit='batch'):
        print(image.shape)
