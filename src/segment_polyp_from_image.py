import tensorflow as tf
from tensorflow.keras.models import load_model


import numpy as np
import cv2
import time

IMG_HEIGHT = 512
IMG_WIDTH = 512


def predict_mask(model, input_img):
    input_img = np.expand_dims(input_img, axis=0)
    print("[info]Inferenceing.....")
    start = time.time()
    pred_mask = model.predict(input_img)[0]
    pred_mask = (pred_mask > 0.5).astype(np.int8)
    pred_mask = np.squeeze(pred_mask)
    pred_mask = [pred_mask, pred_mask, pred_mask]
    pred_mask = np.transpose(pred_mask, (1, 2, 0))

    return pred_mask


def main():
    model = loadModel('Trained model/unetBN_BGR.h5')
    input_img = read_image('sampleImages/polyp1.jpg')
    mask_pred = predict_mask(model, input_img)
    print(mask_pred[0, :, :])
    #cv2.imshow('input image',input_img)
    cv2.imshow('predicted mask', mask_pred)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
