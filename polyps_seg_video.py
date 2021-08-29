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

import cv2
import numpy as np
import argparse
from time import time

import tensorflow as tf
from tensorflow.keras import models



def preprocess_input(x: np.ndarray, input_size: int = 352) -> tf.Tensor:
	assert isinstance(x, np.ndarray)

	# convert to tf.Tensor float32 from ndarray uint8
	x = tf.convert_to_tensor(x, dtype=tf.float32)
	x = tf.image.resize(x, [input_size, input_size])
	x = tf.expand_dims(x, axis=0)
	x = x/255.0

	return x


def process_output(raw_out: tf.Tensor, image_h: int, image_w: int, threshold: float = 0.5) -> np.ndarray:
	assert isinstance(raw_out, tf.Tensor)

	res = tf.sigmoid(raw_out)
	res = tf.cast(tf.math.greater(res, threshold), dtype=tf.uint8)
	res = tf.squeeze(res, axis = 0)
	res = tf.image.resize(res, [image_h, image_w])
	res = res * 255
	res = res.numpy()

	return res


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


def run(
        video_path: str,
        model_path: str,
        input_size: int = 352,
        threshold: float = 0.5
):
    assert os.path.isfile(video_path)
    assert os.path.exists(model_path)
    assert 1.0 > threshold > 0.0

    pranet = get_model(model_path=model_path) # load model

    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(video_path)

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        # Get frame rate information
        # Obtain frame size information using get() method
        frame_width = int(vid_capture.get(3))
        frame_height = int(vid_capture.get(4))
        frame_size = (frame_width, frame_height)
        fps = vid_capture.get(5)
        print('Frames per second : ', fps, 'FPS')

        output = cv2.VideoWriter(
            f'output_video_from_file_{time()}.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            # cv2.imshow('Frame',frame)
            # key = cv2.waitKey()
            input_frame = preprocess_input(frame, input_size=input_size)

            start = time()
            outs = pranet(input_frame)
            end = time()

            res = outs[-1]
            res = process_output(raw_out=res, image_h=frame_height, image_w=frame_width, threshold=threshold)

            final_frame = ((0.5 * frame) + (0.5 * res)).astype("uint8")
            
            elaps_time = (end - start)*1000
            tf.print(f"Inference time for one frame: {elaps_time}ms")

            output.write(final_frame)
        else:
            break

    # Release the video capture object
    vid_capture.release()


if __name__ == "__main__":
    __description__ = '''
    Python script to segment polyp in real time and to output file
    '''
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--model_path', type=str,
                        help='path to savedModel', required=True)
    parser.add_argument('--video_path', type=str, 
                        help='path to video', required=True)
    parser.add_argument('--inputsize', type=int,
                        default=352, help='input image size')
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='threshold for mask')

    opt = parser.parse_args()
    run(
        video_path=opt.video_path,
        model_path=opt.model_path,
        input_size=opt.inputsize,
        threshold=opt.threshold
    )
