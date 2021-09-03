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


#test
if __name__ == "__main__":
    from visualize_bce_iou_loss_weigth import read_mask

    path_to_mask1 = "polyps_dataset/masks/cjxn4fm0wg1cn0738rvy81d2v.jpg"
    path_to_mask2 = "polyps_dataset/masks/cju0qkwl35piu0993l0dewei2.jpg"

    loss_w_bce_dice = WBCEDICELoss(name='structure_loss')

    y_mask = read_mask(path_to_mask1)
    y_pred = read_mask(path_to_mask2)

    total_w_bce_dice_loss = loss_w_bce_dice(y_mask, y_pred)

    tf.print(f"w_bce_dice_loss: {total_w_bce_dice_loss}")
