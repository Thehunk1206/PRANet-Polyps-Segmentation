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
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as dtedt

class WBCEDICELoss(tf.keras.losses.Loss):
    def __init__(self, name: str,):
        super(WBCEDICELoss, self).__init__(name=name)

    @tf.function
    def call(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        bce_iou_weights = 1 + 5 * \
            tf.abs(tf.nn.avg_pool2d(y_mask, ksize=31,
                   strides=1, padding="SAME")-y_mask)

        # weighted BCE loss
        bce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(y_mask, y_pred)
        wbce_loss = tf.reduce_sum(
            bce_loss*bce_iou_weights, axis=(1, 2)) / tf.reduce_sum(bce_iou_weights, axis=(1, 2))

        # weighted DICE loss
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)

        inter = tf.reduce_sum((y_pred * y_mask) * bce_iou_weights, axis=(1, 2))
        union = tf.reduce_sum((y_pred + y_mask) * bce_iou_weights, axis=(1, 2))
        wdice_loss = 1 - ((2*inter) / union+1e-15)

        weighted_bce_dice_loss = tf.reduce_mean(
            wbce_loss + wdice_loss)
        return weighted_bce_dice_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)



def dice_coef(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Sorensen Dice coeffient.
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: Dice coeff value ranging between [0-1]
    '''
    smooth = 1e-15
    y_pred = tf.sigmoid(y_pred)

    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=( 1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = tf.reduce_mean(((2*intersection+smooth) / union))

    return dice

def iou_metric(y_mask: tf.Tensor, y_pred: tf.Tensor)-> tf.Tensor:
    '''
    Intersection over Union measure
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: IoU measure value ranging between [0-1]
    '''
    smooth = 1e-15

    y_pred = tf.sigmoid(y_pred)
    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)


    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=( 1, 2))
    
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth

    iou = tf.reduce_mean((intersection)/(union-intersection))

    return iou

# All other metric (wFb, Sα, Emaxφ, MAE) 
class WFbetaMetric(object):
    def __init__(self, beta:int = 1) -> None:
        super().__init__()
        self.beta = beta
        self.eps = 1e-12

    def _gaussian_distribution(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        It returns the gassian distribution of the given ndarray
        args:
            [x] - [ndarray] 
            mu - [float] mean of the gaussian distribution
            sigma - [float] standard deviation of the gaussian distribution
        return:
            ndarray - Gaussian distribution of the given x ndarray with
            standard deviation sigma and mean mu
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -np.power(
                (x - mu) / sigma, 2) / 2)

    def _generate_gaussian_kernel(self, size: int, sigma: float = 1.0, mu: float = 0.0) -> np.ndarray:
        """
        Generate gaussian kernel of given given size and dims (sizexsize)
        args:
            size - [int] deifnes the size of the kernel (sizexsize)
            sigma - [float] standard diviation of gaussian
                    distribution. It cannot be 0.0
            mu - [float] mean of the gaussian distribution
        return:
            kernel2D - [ndarray] gaussian kernel the values are in range (0,1)
        """
        # create the 1D array of equally spaced distance point of given size
        self.kernel_1d = np.linspace(-(size//2), size//2, size)
        # get the gaussian distribution of the 1D array
        self.kernel_1d = self._gaussian_distribution(
            self.kernel_1d, mu, sigma)

        # Compute the outer product of kernel1D tranpose and kernel1D
        self.kernel_2d = np.outer(self.kernel_1d.T, self.kernel_1d)
        # normalize the the outer product to suish the values between 0.0-1.0
        self.kernel_2d *= 1.0/self.kernel_2d.max()
        return self.kernel_2d


    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        '''
        Rerefence https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency/blob/master/saliency_metric.py
        The following metric is from paper: How to Evaluate Foreground Maps? (CVPR2014)
        '''
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)
        y_pred = tf.squeeze(y_pred)

        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.int32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.int32)
        y_pred = y_pred.numpy()
        y_mask = y_mask.numpy()

        Dst, Idxt = dtedt(y_mask == 0, return_indices=True)
        
        E = np.abs(y_pred - y_mask)

        Et = np.copy(E)
        Et[y_mask == 0] = Et[Idxt[0][y_mask == 0], Idxt[1][y_mask == 0]]

        K = self._generate_gaussian_kernel(size=7, sigma=5.0)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(y_mask & (EA < E), EA, E)

        B = np.where(y_mask == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(y_mask))
        Ew = MIN_E_EA * B

        TPw = np.sum(y_mask) - np.sum(Ew[y_mask == 1])
        FPw = np.sum(Ew[y_mask == 0])

        R = 1 - np.mean(Ew[y_mask])
        P = TPw / (self.eps + TPw + FPw)

        wfb = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return wfb

#test
if __name__ == "__main__":
    from visualize_bce_iou_loss_weigth import read_mask

    path_to_mask1 = "polyps_dataset/masks/cjxn4fm0wg1cn0738rvy81d2v.jpg"
    path_to_mask2 = "polyps_dataset/masks/cju0qkwl35piu0993l0dewei2.jpg"

    loss_w_bce_dice = WBCEDICELoss(name='structure_loss')
    wFb_metric = WFbetaMetric()

    y_mask = read_mask(path_to_mask1)
    y_pred = read_mask(path_to_mask2)

    # y_mask = tf.random.normal([8, 352, 352, 1])
    # y_pred = tf.random.normal([8, 352, 352, 1])

    total_w_bce_dice_loss = loss_w_bce_dice(y_mask, y_pred)
    dice_metric = dice_coef(y_mask, y_pred)
    iou = iou_metric(y_mask, y_pred)
    wfb = wFb_metric(y_mask=y_mask, y_pred=y_pred)

    tf.print(f"w_bce_dice_loss: {total_w_bce_dice_loss}")
    tf.print(f"dice coef: {dice_metric}")
    tf.print(f"IoU: {iou}")
    tf.print(f"wFbeta: {wfb}")

