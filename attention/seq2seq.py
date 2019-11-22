from functools import partial

import tensorflow as tf

layers = tf.keras.layers


class _Seq2SeqBase(object):
    @staticmethod
    def gru():
        return layers.CuDNNGRU if tf.test.is_gpu_available() else layers.GRU

    @staticmethod
    def lstm():
        return layers.CuDNNLSTM if tf.test.is_gpu_available() else layers.LSTM


class RNNEncoder(_Seq2SeqBase):
    def __init__(self, units, bidirectional=False, merge_mode=None):
        rnn_model = partial(self.gru(), units=units, return_sequences=True, return_state=True, unroll=True)
        self.forward_rnn = rnn_model(go_backwards=False, name='enc_forward_rnn')
        self.backward_rnn = rnn_model(go_backwards=True, name='enc_backward_rnn') if bidirectional else None
        self.merge_mode = merge_mode

    def __call__(self, inputs):
        forward_results = self.forward_rnn(inputs)
        if self.backward_rnn:
            backward_results = self.backward_rnn(inputs)
            if not self.merge_mode:
                # follow Bahdanau's paper
                backward_results[0] = layers.Concatenate()([forward_results[0], backward_results[0]])
                final_results = backward_results
            else:
                merge_func = layers.Concatenate() if self.merge_mode == 'concat' else layers.Add()
                final_results = [merge_func([i, j]) for i, j in zip(forward_results, backward_results)]
        else:
            final_results = forward_results
        output, hidden = final_results[0], final_results[1:]
        hidden = [layers.Dense(units=self.forward_rnn.units, activation='tanh')(x) for x in hidden]
        return output, hidden


class RNNWithAttentionDecoder(_Seq2SeqBase):
    def __init__(self, units, n_classes, dec_max_time_steps, eos_token=0,
                 attn_method='concat', attn_before_rnn=True, **kwargs):
        self.rnn = self.gru()(units=units, return_state=True)
        self.attn_score = self.build_attn_score_func(units, attn_method, **kwargs)
        self.attn_combine = layers.Dense(units=units, activation='tanh', name='dec_attn_combine')
        self.attn_before_rnn = attn_before_rnn
        self.output_fc = layers.Dense(units=n_classes, name='dec_output_fc')
        self.dec_max_time_steps = dec_max_time_steps
        self.eos_token = eos_token  # todo: early stopping

    @staticmethod
    def build_attn_score_func(units, attn_method, **kwargs):  # todo: share?
        if attn_method == 'concat':
            fcs = [
                tf.layers.Dense(units=units, activation='tanh', name='w'),
                tf.layers.Dense(units=1, name='r')
            ]

            def f(*args):
                _, h, e = args
                h = tf.expand_dims(h, axis=1)  # ?*1*N
                h = tf.tile(h, multiples=[1, e.shape[1], 1])  # ?*20*N
                x = tf.concat([e, h], axis=-1)
                for layer in fcs:
                    x = layer(x)
                return x  # ?*20*1

            return f
        elif attn_method == 'location':
            enc_max_time_steps = kwargs.get('enc_max_time_steps', None)
            assert enc_max_time_steps
            fc = tf.layers.Dense(units=enc_max_time_steps)

            def f(*args):
                x = fc(tf.concat(args[:-1], axis=-1))  # ?*20
                return tf.expand_dims(x, axis=-1)  # ?*20*1

            return f
        elif attn_method == 'dot':
            def f(*args):
                _, h, e = args
                h = tf.expand_dims(h, axis=-1)  # ?*32*1
                return tf.matmul(e, h)  # ?*20*1

            return f
        else:
            raise NotImplemented

    def __call__(self, inputs, encoder_output, encoder_state, teacher_forcing, **kwargs):
        hidden_state = encoder_state
        outputs = []

        def without_teacher_forcing():
            embed = kwargs.get('embed', None)
            assert embed
            return embed(tf.argmax(pred, axis=1))

        for step in range(self.dec_max_time_steps):
            if step == 0:
                x = inputs[:, 0, :]
            else:
                x = tf.cond(teacher_forcing, true_fn=lambda: inputs[:, step, :],
                            false_fn=without_teacher_forcing, name='dec_switch_teacher_forcing')
            '''calculate attention'''
            h_state = hidden_state[0]
            atten_scores = self.attn_score(x, h_state, encoder_output)
            atten_weights = tf.nn.softmax(atten_scores, dim=1)
            atten_context = tf.multiply(encoder_output, atten_weights)  # ?*20*32 ?*20*1
            atten_context = tf.reduce_sum(atten_context, axis=1)
            '''across rnn'''
            if self.attn_before_rnn:
                x = tf.expand_dims(tf.concat([atten_context, x], axis=-1), axis=1)  # todo: delete x?
                results = self.rnn(x, initial_state=hidden_state)
                output, hidden_state = results[0], results[1:]
            else:
                # follow Luong's paper. a little bit different~
                x = tf.expand_dims(x, axis=1)
                results = self.rnn(x, initial_state=hidden_state)
                output, hidden_state = results[0], results[1:]
                x = tf.concat([atten_context, output], axis=-1)
                output = self.attn_combine(x)
            pred = self.output_fc(output)  # logits
            outputs.append(pred)

        outputs = tf.stack(outputs, axis=1)
        return outputs


def _default_batchify_fn(data):
    if isinstance(data[0], np.ndarray):
        return np.stack(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [_default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return data


class _MMetric(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def update(self, num, total):
        self.num += num
        self.total += total

    def get(self):
        return self.num / self.total

    def reset(self):
        self.num = 0
        self.total = 0


if __name__ == '__main__':
    import warnings
    import os
    import numpy as np
    import pandas as pd
    from mxnet.gluon.data import ArrayDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    hidden_size = 32
    sos_token = 10
    use_teacher_forcing_ratio = 0.5
    '''build encoder'''
    encoder_input = tf.placeholder(tf.int32, shape=(None, 20))
    encoder_embedding = layers.Embedding(input_dim=11, output_dim=8, trainable=True)
    encoder = RNNEncoder(units=hidden_size, bidirectional=True, merge_mode='sum')
    encoder_output, encoder_state = encoder(inputs=encoder_embedding(encoder_input))
    '''build decoder'''
    decoder_input = tf.placeholder(tf.int32, shape=(None, None))
    teacher_forcing = tf.placeholder_with_default(False, shape=None)
    decoder = RNNWithAttentionDecoder(
        units=hidden_size,
        n_classes=10,
        enc_max_time_steps=20,
        dec_max_time_steps=20,
        attn_method='dot',
        attn_before_rnn=False
    )
    decoder_output = decoder(inputs=encoder_embedding(decoder_input), encoder_output=encoder_output,
                             encoder_state=encoder_state, teacher_forcing=teacher_forcing,
                             embed=encoder_embedding)
    softmax_label = tf.placeholder(tf.int64, shape=(None, 20))

    '''build loss'''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_output, labels=softmax_label)
    loss = tf.reduce_mean(loss)
    '''build optimizer'''
    opt = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)
    '''build metric'''
    pred_label = tf.argmax(decoder_output, axis=-1)
    n_true = tf.reduce_all(tf.equal(pred_label, softmax_label), axis=1)
    n_true = tf.cast(n_true, dtype=tf.int32)
    n_true = tf.reduce_sum(n_true)

    '''load data'''
    def load_data(path):
        return pd.read_csv(path, header=None).values
    X_train = load_data('./dataset/task8_train_input.csv')
    y_train = load_data('./dataset/task8_train_output.csv')
    X_test = load_data('./dataset/task8_test_input.csv')
    y_test = load_data('./dataset/task8_test_output.csv')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=0)
    print('TrainSet Shape:{}'.format(X_train.shape))
    print('TestSet Shape:{}'.format(X_test.shape))
    build_dataloader = partial(DataLoader, batch_size=32, shuffle=False, last_batch='keep',
                               batchify_fn=_default_batchify_fn)
    train_dataloader = build_dataloader(dataset=ArrayDataset(X_train, y_train))
    test_dataloader = build_dataloader(dataset=ArrayDataset(X_test, y_test))
    val_dataloader = build_dataloader(dataset=ArrayDataset(X_val, y_val))

    '''start training'''
    sess.run(tf.global_variables_initializer())
    train_loss, train_acc = _MMetric(), _MMetric()
    print_freq = 50
    for step, (x, y) in enumerate(tqdm(train_dataloader, desc='Training', position=0)):
        sos_input = np.ones(shape=(len(y), 1), dtype=np.int32) * sos_token
        t = np.random.rand() < use_teacher_forcing_ratio
        d = sos_input if not t else np.concatenate((sos_input, y[:, 1:]), axis=1)
        feed_dict = {encoder_input: x, decoder_input: d, softmax_label: y, teacher_forcing: t}
        _, loss_value, n_true_value = sess.run([opt, loss, n_true], feed_dict=feed_dict)
        train_loss.update(loss_value, 1)
        train_acc.update(n_true_value, len(x))

        if step != 0 and step % print_freq == 0:
            '''Evaluate on validation set'''
            val_loss, val_acc = _MMetric(), _MMetric()
            for x, y in val_dataloader:
                sos_input = np.ones(shape=(len(y), 1), dtype=np.int32) * sos_token
                feed_dict = {encoder_input: x, decoder_input: sos_input, softmax_label: y, teacher_forcing: False}
                loss_value, n_true_value = sess.run([loss, n_true], feed_dict=feed_dict)
                val_loss.update(loss_value, 1)
                val_acc.update(n_true_value, len(x))
            tqdm.write(
                '[Step {}/{}] train-loss: {}, train-acc: {} val-loss: {}, val-acc: {}'.format(
                    step, len(train_dataloader), train_loss.get(), train_acc.get(), val_loss.get(), val_acc.get()
                )
            )
            train_loss.reset()
            train_acc.reset()

    '''start testing'''
    test_loss, test_acc = _MMetric(), _MMetric()
    for step, (x, y) in enumerate(tqdm(test_dataloader, desc='Testing', position=0)):
        sos_input = np.ones(shape=(len(y), 1), dtype=np.int32) * sos_token
        feed_dict = {encoder_input: x, decoder_input: sos_input, softmax_label: y, teacher_forcing: False}
        loss_value, n_true_value = sess.run([loss, n_true], feed_dict=feed_dict)
        test_loss.update(loss_value, 1)
        test_acc.update(n_true_value, len(x))
        if step != 0 and step % print_freq == 0:
            tqdm.write('[Step {}/{}] test-loss: {}, test-acc: {}'
                       .format(step, len(test_dataloader), test_loss.get(), test_acc.get()))
    tqdm.write('[final] test-loss: {}, test-acc: {}'.format(test_loss.get(), test_acc.get()))
