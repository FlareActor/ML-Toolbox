"""Custom loss function
"""
import keras.backend as K
import tensorflow as tf
import math
from collections import Iterable
from keras import initializers
from keras.engine.topology import Layer


def focal_loss(y_true, y_pred, alpha=1.5, gamma=3):
    """Focal loss"""
    old_loss = K.categorical_crossentropy(y_true, y_pred)  # 普通交叉熵
    sample_weight = K.pow(1 - K.sum(y_pred * y_true, axis=1), gamma)  # 样本损失权重
    return alpha * sample_weight * old_loss


class AngularMargin(Layer):
    """Additive angular margin loss"""

    def __init__(self, output_dim, s=64, m=0.5, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.mask_m = False
        super(AngularMargin, self).__init__(**kwargs)

    def build(self, input_shape):
        """定义权重"""
        nb_feats = input_shape[0][-1] if isinstance(
            input_shape[0], Iterable) else input_shape[-1]
        self.W = self.add_weight((nb_feats, self.output_dim), initializer=self.init,
                                 name='{}_W'.format(self.name), trainable=True)
        super(AngularMargin, self).build(input_shape)

    def call(self, inputs, mask=None):
        l2_norm = K.l2_normalize(inputs[0], axis=1)  # 特征归一化
        self.W = K.l2_normalize(self.W, axis=0)  # 权重归一化
        cos_theta = K.dot(l2_norm, self.W)
        if self.mask_m:
            # 不考虑角度间隔
            return K.softmax(cos_theta * self.s)
        # 求sin theta，反向传播求导时分母为0
        sin_theta = 1. - K.square(cos_theta)
        epsilon = K.zeros_like(sin_theta) + K.epsilon()
        sin_theta = tf.where(sin_theta > 0, sin_theta, epsilon)
        sin_theta = K.sqrt(sin_theta)
        # cos(theta+m)=cos(theta)*cos(m)-sin(theta)*sin(m)
        cos_theta_add_m = cos_theta * \
            math.cos(self.m) - sin_theta * math.sin(self.m)
        #
        mm = math.sin(math.pi - self.m) * self.m
        threshold = math.cos(math.pi - self.m)
        cond = cos_theta - threshold
        cond = K.cast(K.relu(cond), dtype=tf.bool)
        keep_val = cos_theta - mm
        cos_tm_temp = tf.where(cond, cos_theta_add_m, keep_val)
        #
        gt_one_hot = inputs[1]
        if gt_one_hot.shape[-1] == 1:
            gt_one_hot = K.cast(gt_one_hot, tf.int32)
            gt_one_hot = K.one_hot(gt_one_hot, self.output_dim)
            gt_one_hot = K.reshape(gt_one_hot, (-1, self.output_dim))
        diff = cos_tm_temp - cos_theta
        diff = tf.multiply(diff, gt_one_hot)
        output = (cos_theta + diff) * self.s
        return K.softmax(output)

    def compute_output_shape(self, input_shape):
        nb_samples = input_shape[0][0] if isinstance(
            input_shape[0], Iterable) else input_shape[0]
        return nb_samples, self.output_dim
