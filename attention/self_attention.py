"""Attention is all u need!
"""
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras import backend as K


class Attention(Layer):
    def __init__(self, dropout=0, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dropout = dropout

    def call(self, x, mask=None):
        q, k, v = x
        dot = K.batch_dot(q, tf.transpose(k, [0, 2, 1]))
        dk = k.shape.as_list()[-1]
        dot /= np.sqrt(dk)
        if mask is not None:
            m = K.cast(mask[1], tf.float32)
            m = (1 - m) * 1e12
            m = K.expand_dims(m, 1)
            dot = dot - m
        p = K.softmax(dot)
        p = K.dropout(p, level=self.dropout)
        return K.batch_dot(p, v)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[-1][-1])

    def compute_mask(self, inputs, mask=None):
        return mask[0]


class PositionEmbedding(Layer):
    def __init__(self, dim=64, mode='sum', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        assert mode in ['sum', 'concat']
        self.dim = dim
        self.mode = mode

    def call(self, x, mask=None):
        if self.mode == 'sum':
            self.dim = int(x.shape[-1])
        d = K.arange(self.dim / 2, dtype=tf.float32)
        d = 1. / K.pow(10000., 2 * d / self.dim)
        d = K.expand_dims(d, 0)
        t = K.ones_like(x[:, :, 0])
        t = K.cumsum(t, axis=-1) - 1
        t = K.expand_dims(t, -1)
        pe = K.dot(t, d)
        sin = K.sin(pe)
        cos = K.cos(pe)
        ###
        sin = K.repeat_elements(sin, 2, axis=-1)
        cos = K.repeat_elements(cos, 2, axis=-1)
        sin_mask = np.zeros(self.dim)
        sin_mask[::2] += 1
        sin_mask = K.constant(sin_mask)
        cos_mask = np.zeros(self.dim)
        cos_mask[1::2] += 1
        cos_mask = K.constant(cos_mask)
        sin = sin * sin_mask
        cos = cos * cos_mask
        pe = sin + cos
        if self.mode == 'sum':
            x = x + pe
        else:
            x = K.concatenate([x, pe], -1)
        return x

    def compute_output_shape(self, input_shape):
        if self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[-1] + self.dim)
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class MultiHeadAttention(Layer):
    """Multi-head dot product attention"""

    def __init__(self, nb_header=8, output_dim=128, dk=None, dropout=0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.nb_header = nb_header
        self.output_dim = output_dim
        self.dk = output_dim // nb_header if dk is None else dk
        self.dropout = dropout

    def build(self, input_shape):
        dim = self.nb_header * self.dk
        self.kernel_q = self.add_weight(name='weight_Q',
                                        shape=(input_shape[0][-1], dim),
                                        initializer='glorot_uniform',
                                        trainable=True)
            self.kernel_k = self.add_weight(name='weight_K',
                                            shape=(input_shape[1][-1], dim),
                                            initializer='glorot_uniform',
                                            trainable=True)
            self.kernel_v = self.add_weight(name='weight_V',
                                            shape=(input_shape[2][-1], dim),
                                            initializer='glorot_uniform',
                                            trainable=True)
            if self.output_dim != dim:
                self.kernel_o = self.add_weight(name='weight_O',
                                                shape=(dim, self.output_dim),
                                                initializer='glorot_uniform',
                                                trainable=True)
            super(MultiHeadAttention, self).build(input_shape)

    def call(self, x, mask=None):
        q, k, v = x
        qw = K.dot(q, self.kernel_q)
        kw = K.dot(k, self.kernel_k)
        vw = K.dot(v, self.kernel_v)
        qw = K.reshape(qw, (-1, qw.shape[1], self.dk, self.nb_header))
        kw = K.reshape(kw, (-1, kw.shape[1], self.dk, self.nb_header))
        vw = K.reshape(vw, (-1, vw.shape[1], self.dk, self.nb_header))
        qw = tf.transpose(qw, [0, 3, 1, 2])
        kw = tf.transpose(kw, [0, 3, 2, 1])
        vw = tf.transpose(vw, [0, 3, 1, 2])
        dot = K.batch_dot(qw, kw) * (self.dk**-0.5)
        if mask is not None:
            m = K.cast(mask[1], tf.float32)
            m = (1 - m) * 1e12
            m = K.expand_dims(m, 1)
            m = K.expand_dims(m, 1)
            dot = dot - m
        p = K.softmax(dot)
        p = K.dropout(p, level=self.dropout)
        o = K.batch_dot(p, vw)
        o = tf.transpose(o, [0, 2, 3, 1])
        o = K.reshape(o, (-1, o.shape[1], o.shape[2] * o.shape[3]))
        if getattr(self, 'kernel_o', None) is not None:
            o = K.dot(o, self.kernel_o)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def compute_mask(self, inputs, mask=None):
        return mask[0]
