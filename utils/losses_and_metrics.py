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


class WBCEIOULoss(tf.keras.losses.Loss):
    def __init__(self, name: str,):
        super(WBCEIOULoss, self).__init__(name=name)

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

        # weighted IOU loss
        y_pred = tf.sigmoid(y_pred)
        inter = tf.reduce_sum((y_pred * y_mask) * bce_iou_weights, axis=(1, 2))
        union = tf.reduce_sum((y_pred + y_mask) * bce_iou_weights, axis=(1, 2))
        wiou_loss = 1 - (inter+1)/(union - inter+1)

        weighted_bce_iou_loss = tf.reduce_mean(
            wbce_loss + wiou_loss)
        return weighted_bce_iou_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, name: str):
        super(SSIMLoss, self).__init__(name=name)
    '''
    Experimental
    '''

    def call(self, y_mask: tf.Tensor, y_pred: tf.Tensor):
        ssim = tf.image.ssim_multiscale(y_mask, y_pred, max_val=1.0)
        return 1 - ssim

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


class DiceCoef(tf.keras.metrics.Metric):

    def __init__(self, name: str, **kwargs):
        super(DiceCoef, self).__init__(name=name, **kwargs)
        self.dice_coef = self.add_weight(
            name='Dice Coefficient', initializer='zeros')

    @tf.function
    def update_state(self, y_mask: tf.Tensor, y_pred: tf.Tensor, **kwargs):
        smooth = 1e-15
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
        intersection = tf.squeeze(tf.reduce_sum(
            tf.multiply(y_mask, y_pred), axis=(1, 2)))
        union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth
        dice = tf.reduce_mean(((2*intersection) / union))

        self.dice_coef.assign(dice)

    def result(self):
        return self.dice_coef

    def reset_states(self):
        self.dice_coef.assign(0.0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


def dice_coef(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = 1e-15
    y_pred = tf.sigmoid(y_pred)
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    intersection = tf.squeeze(tf.reduce_sum(
        tf.multiply(y_mask, y_pred), axis=(1, 2)))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth
    dice = tf.reduce_mean(((2*intersection) / union))

    return dice


if __name__ == "__main__":
    from visualize_bce_iou_loss_weigth import read_mask

    path_to_mask1 = "polyps_dataset/masks/cjxn4fm0wg1cn0738rvy81d2v.jpg"
    path_to_mask2 = "polyps_dataset/masks/cju0qkwl35piu0993l0dewei2.jpg"

    loss_w_bce_iou = WBCEIOULoss(name='structure_loss')
    loss_ms_ssim = SSIMLoss(name='SSIM_loss')
    # dice_metric = DiceCoef(name='dice metric')

    y_mask = read_mask(path_to_mask1)
    y_pred = read_mask(path_to_mask2)

    # y_mask = tf.random.normal([8, 352, 352, 1])
    # y_pred = tf.random.normal([8, 352, 352, 1])

    total_w_bce_iou_loss = loss_w_bce_iou(y_mask, y_pred)
    total_ssim_loss = loss_ms_ssim(y_mask, y_pred)
    # dice_metric.update_state(y_mask, y_pred)
    dice_metric = dice_coef(y_mask, y_pred)

    print(f"w_bce_iou_loss: {total_w_bce_iou_loss}")
    print(f"SSIM loss: {total_ssim_loss}")
    print(f"dice coef: {dice_metric}")
