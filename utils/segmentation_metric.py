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
from scipy.ndimage import convolve, center_of_mass, distance_transform_edt as dtedt


def dice_coef(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Sorensen Dice coeffient.
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: Dice coeff value ranging between [0-1]
    '''
    smooth = 1e-15

    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = tf.reduce_mean(((2*intersection+smooth) / union))

    return dice


def iou_metric(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Intersection over Union measure
    args:   y_mask->tf.Tensor  Ground truth Map
            y_pred->tf.Tensor  Computed Raw Mask

    return: IoU measure value ranging between [0-1]
    '''
    smooth = 1e-15

    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    intersection = tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2))

    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth

    iou = tf.reduce_mean((intersection)/(union-intersection))

    return iou

def MAE(y_mask: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(tf.abs(y_pred - y_mask))

# All other metric (wFb, Sα, Emaxφ, MAE)

class WFbetaMetric(object):
    '''
    Rerefence https://github.com/DengPingFan/PraNet/tree/master/eval
    The following metric is from paper: How to Evaluate Foreground Maps? (CVPR2014)
    '''
    def __init__(self, beta: int = 1) -> None:
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
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)  # (b,h,w,c) => (h,w)
        y_pred = tf.squeeze(y_pred)  # (b,h,w,c) => (h,w)

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

        B = np.where(y_mask == 0, 2 - np.exp(np.log(0.5) /
                     5 * Dst), np.ones_like(y_mask))
        Ew = MIN_E_EA * B

        TPw = np.sum(y_mask) - np.sum(Ew[y_mask == 1])
        FPw = np.sum(Ew[y_mask == 0])

        R = 1 - np.mean(Ew[y_mask])
        P = TPw / (self.eps + TPw + FPw)

        wfb = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return tf.cast(wfb, dtype=tf.float32)


class SMeasure(object):
    '''
    Rerefence https://github.com/DengPingFan/PraNet/tree/master/eval
    Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    '''
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
    
    def _object(self, inp1:np.ndarray, inp2:np.ndarray)->tf.Tensor:
        '''
        Computes BG and FG comparison of GT and SM
        '''
        x = np.mean(inp1[inp2])
        sigma_x = np.std(inp1[inp2])
        score = 2 * x / (x**2 + 1 + sigma_x + 1e-8)
        return tf.cast(score,dtype=tf.float32)

    def s_object(self, SM: tf.Tensor, GT: tf.Tensor)->tf.Tensor:
        '''
        Computes similarity between GT and SM at object level
        '''
        fg = SM * GT
        bg = (1 - SM) * (1 - GT)

        u = tf.reduce_mean(GT)
        # converting GT to logical type(i.e bool type)
        GT = tf.cast(GT, dtype=tf.bool)
        return u * self._object(fg.numpy(), GT.numpy()) + (1 - u) * self._object(bg.numpy(), tf.logical_not(GT.numpy()))

    def _ssim(self, SM: tf.Tensor, GT: tf.Tensor):
        h, w = SM.shape
        N = h * w

        x = tf.reduce_mean(SM)
        y = tf.reduce_mean(GT)

        sigma_x = tf.math.reduce_variance(SM)
        sigma_y = tf.math.reduce_variance(GT)
        sigma_xy = tf.reduce_sum((SM - x) * (GT - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

    def _divideGT(self, GT: tf.Tensor, x: int, y: int) -> tuple:
        # get H.W and area of GT
        h, w = GT.shape
        area = h*w
        # divide GT into four blocks
        UL = GT[0:y, 0:x]
        UR = GT[0:y, x:w]
        LL = GT[y:h, 0:x]
        LR = GT[y:h, x:w]

        # calculate weigths i.e  area_of_blocks/area_of_GT
        w1 = (x * y) / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return UL, UR, LL, LR, w1, w2, w3, w4

    def _divideSM(self, SM: tf.Tensor, x: int, y: int) -> tuple:
        # get H.W and area of SM
        h, w = SM.shape
        area = h*w
        # divide SM into four blocks
        UL = SM[0:y, 0:x]
        UR = SM[0:y, x:w]
        LL = SM[y:h, 0:x]
        LR = SM[y:h, x:w]

        return UL, UR, LL, LR

    def s_region(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        '''
        Calculates Region aware structural similarity 
        '''
        # get the centre of mass of y_mask
        [y, x] = center_of_mass(y_mask.numpy())
        x = int(round(x)) + 1
        y = int(round(x)) + 1

        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(
            GT=y_mask, x=x, y=y)
        sm1, sm2, sm3, sm4 = self._divideSM(SM=y_pred, x=x, y=y)

        score1 = self._ssim(sm1, gt1)
        score2 = self._ssim(sm2, gt2)
        score3 = self._ssim(sm3, gt3)
        score4 = self._ssim(sm4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)  # (b,h,w,c) => (h,w)
        y_pred = tf.squeeze(y_pred)  # (b,h,w,c) => (h,w)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)
        y = tf.reduce_mean(y_mask)
        if y == 0:
            score = 1 - tf.reduce_mean(y_pred)
        elif y == 1:
            score = tf.reduce_mean(y_pred)
        else:
            score = self.alpha * self.s_object(y_pred, y_mask) + (1 - self.alpha) * self.s_region(y_mask=y_mask,y_pred=y_pred)
        return score


class Emeasure(object):
    ''' 
    Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    Reference: https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency/blob/master/saliency_metric.py
    '''
    def __init__(self):
        pass

    def AlignmentTerm(self, dFM, dy_mask):
        mu_FM = np.mean(dFM)
        mu_y_mask = np.mean(dy_mask)
        align_FM = dFM - mu_FM
        align_y_mask = dy_mask - mu_y_mask
        align_Matrix = 2. * (align_y_mask * align_FM) / \
            (align_y_mask * align_y_mask + align_FM * align_FM + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)  # (b,h,w,c) => (h,w)
        y_pred = tf.squeeze(y_pred)  # (b,h,w,c) => (h,w)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

        y_mask = y_mask.numpy()
        y_pred = y_pred.numpy()

        th = 2 * y_pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(y_mask.shape)
        FM[y_pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        y_mask = np.array(y_mask, dtype=bool)
        dFM = np.double(FM)

        if (sum(sum(np.double(y_mask))) == 0):
            enhanced_matrix = 1.0-dFM
        elif (sum(sum(np.double(~y_mask))) == 0):
            enhanced_matrix = dFM
        else:
            dy_mask = np.double(y_mask)
            align_matrix = self.AlignmentTerm(dFM, dy_mask)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)

        [w, h] = np.shape(y_mask)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)

        return tf.cast(score, dtype=tf.float32)



# test 
if __name__ == "__main__":
    from visualize_bce_iou_loss_weigth import read_mask

    path_to_mask1 = "polyps_dataset/masks/cjxn4fm0wg1cn0738rvy81d2v.jpg"
    path_to_mask2 = "polyps_dataset/masks/cju0qkwl35piu0993l0dewei2.jpg"

    y_mask = read_mask(path_to_mask1)
    y_pred = read_mask(path_to_mask2)

    wFb_metric = WFbetaMetric()
    smeasure_metric = SMeasure()
    emeasure_metric = Emeasure()


    dice_metric = dice_coef(y_mask, y_pred)
    iou = iou_metric(y_mask, y_pred)
    wfb = wFb_metric(y_mask=y_mask, y_pred=y_pred)
    smeasure = smeasure_metric(y_mask=y_mask, y_pred=y_pred)
    emeasure = emeasure_metric(y_mask=y_mask, y_pred=y_pred)
    mae = MAE(y_mask=y_mask, y_pred=y_mask)

    tf.print(f"dice coef: {dice_metric}")
    tf.print(f"IoU: {iou}")
    tf.print(f"wFbeta: {wfb}")
    tf.print(f"Smeasure: {smeasure}")
    tf.print(f"Emeasure: {emeasure}")
    tf.print(f"mae: {mae}")


