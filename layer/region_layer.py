import tensorflow as tf
import numpy as np


class RegionLayer(object):
    def __init__(self, nb_rows=4, nb_cols=4, func=None, training=False):
        self.nb_rows = int(nb_rows)
        self.nb_cols = int(nb_cols)
        self.func = self._default_func if func is None else func
        self.training = training  # For BN

    def __call__(self, layer):
        feat_map_size = layer.get_shape().as_list()[1:3]
        region_width = feat_map_size[1] // self.nb_cols
        region_height = feat_map_size[0] // self.nb_rows
        rows = []
        for i in range(self.nb_rows):
            row = []
            for j in range(self.nb_cols):
                top = i * region_height
                left = j * region_width
                region = tf.strided_slice(layer,
                                          begin=[0, top, left, 0],
                                          end=[10000, top + region_height,
                                               left + region_width, 1000],
                                          strides=[1, 1, 1, 1])
                row.append(self.func(region))
            rows.append(tf.concat(row, axis=2))
        regions = tf.concat(rows, axis=1)
        return regions

    def _default_func(self, region):
        nb_channels = region.get_shape().as_list()[-1]
        with tf.name_scope('region_layer'):
            shortcut = region
            x = tf.layers.batch_normalization(
                region, axis=-1, training=self.training)
            x = tf.nn.relu(x)
            initial = tf.truncated_normal(
                [3, 3, nb_channels, nb_channels], stddev=0.01)
            W = tf.Variable(initial)
            initial = tf.constant(1., shape=[nb_channels])
            b = tf.Variable(initial)
            x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
            x = x + shortcut
        return x


if __name__ == '__main__':
    # test
    with tf.Session():
        i = tf.constant(shape=(1, 8, 8, 1), value=np.array(
            range(64)).reshape((1, 8, 8, 1)))
        x = RegionLayer(nb_cols=2, nb_rows=2, func=lambda x: x * 2)(i)
        x.eval()[:, :, :, 0]
