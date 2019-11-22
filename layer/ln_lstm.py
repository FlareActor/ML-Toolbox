"""Combine layer normalization with lstm
"""
import keras.backend as K
from keras.layers.recurrent import LSTMCell, LSTM


def _generate_dropout_mask(ones, rate, training=None, count=1):
    # drop inputs only during training stage
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


def _ln(x, gamma, beta, eps=1e-5):
    _mean = K.mean(x, axis=-1, keepdims=True)
    _std = K.sqrt(K.var(x, axis=-1, keepdims=True) + eps)
    x_norm = (x - _mean) / _std
    x_scale = x_norm * gamma + beta
    return x_scale


class LNLSTMCell(LSTMCell):
    def build(self, input_shape):
        super(LNLSTMCell, self).build(input_shape)
        # add two parameters about layer normalization
        self.ln_gamma = self.add_weight(shape=(self.units * 4,),
                                        name='gamma2scale',
                                        initializer='ones',
                                        trainable=True)
        self.ln_beta = self.add_weight(shape=(self.units * 4,),
                                       name='beta2bias',
                                       initializer='zeros',
                                       trainable=True)
        self.ln_gamma_i = self.ln_gamma[:self.units]
        self.ln_gamma_f = self.ln_gamma[self.units: self.units * 2]
        self.ln_gamma_c = self.ln_gamma[self.units * 2: self.units * 3]
        self.ln_gamma_o = self.ln_gamma[self.units * 3:]

        self.ln_beta_i = self.ln_beta[:self.units]
        self.ln_beta_f = self.ln_beta[self.units: self.units * 2]
        self.ln_beta_c = self.ln_beta[self.units * 2: self.units * 3]
        self.ln_beta_o = self.ln_beta[self.units * 3:]
        # print(self.ln_gamma.shape)  # (64,)

    def call(self, inputs, states, training=None):
        # print(len(states))  # 2
        # print(inputs.shape)  # (?,64)
        # print(states[0].shape) # (?,32)
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)  # (?,64)*(64,32)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            # update gate
            x = x_i + K.dot(h_tm1_i, self.recurrent_kernel_i)
            x = _ln(x, self.ln_gamma_i, self.ln_beta_i)
            i = self.recurrent_activation(x)
            # forget gate
            x = x_f + K.dot(h_tm1_f, self.recurrent_kernel_f)
            x = _ln(x, self.ln_gamma_f, self.ln_beta_f)
            f = self.recurrent_activation(x)
            # calculate new cell state
            x = x_c + K.dot(h_tm1_c, self.recurrent_kernel_c)
            x = _ln(x, self.ln_gamma_c, self.ln_beta_c)
            c = f * c_tm1 + i * self.activation(x)
            # output gate
            x = x_o + K.dot(h_tm1_o, self.recurrent_kernel_o)
            x = _ln(x, self.ln_gamma_o, self.ln_beta_o)
            o = self.recurrent_activation(x)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            # z = _ln(z, self.ln_gamma, self.ln_beta)
            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(_ln(z0, self.ln_gamma_i, self.ln_beta_i))
            f = self.recurrent_activation(_ln(z1, self.ln_gamma_f, self.ln_beta_f))
            c = f * c_tm1 + i * self.activation(_ln(z2, self.ln_gamma_c, self.ln_beta_c))
            o = self.recurrent_activation(_ln(z3, self.ln_gamma_o, self.ln_beta_o))

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]


class LNLSTM(LSTM):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        super(LNLSTM, self).__init__(units,
                                     activation=activation,
                                     recurrent_activation=recurrent_activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     recurrent_initializer=recurrent_initializer,
                                     bias_initializer=bias_initializer,
                                     unit_forget_bias=unit_forget_bias,
                                     kernel_regularizer=kernel_regularizer,
                                     recurrent_regularizer=recurrent_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     recurrent_constraint=recurrent_constraint,
                                     bias_constraint=bias_constraint,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout,
                                     implementation=implementation,
                                     return_sequences=return_sequences,
                                     return_state=return_state,
                                     go_backwards=go_backwards,
                                     stateful=stateful,
                                     unroll=unroll,
                                     **kwargs
                                     )
        self.cell = LNLSTMCell(units,
                               activation=activation,
                               recurrent_activation=recurrent_activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               unit_forget_bias=unit_forget_bias,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer,
                               bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint,
                               recurrent_constraint=recurrent_constraint,
                               bias_constraint=bias_constraint,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout,
                               implementation=implementation)
