import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# ===========custom loss==============================


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# =========load model using custom loss===============
def loadModel(model_path):
    print("[info] loading model......")
    unet_bn = load_model(model_path, custom_objects={
        'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})
    return unet_bn

# ======reading image from path===============


def read_image(path, IMG_HEIGHT, IMG_WIDTH):
    '''Read the image from the given path and normalize the pixel values'''
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img/255.0
    return img
