import tensorflow as tf
from tensorflow.python.framework import graph_util

if __name__ == '__main__':
    saver = tf.train.import_meta_graph('./model/tf/checkpoint.ckpt.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, './model/tf/checkpoint.ckpt')
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['fc1/add'])
        with tf.gfile.FastGFile('./model/tf/checkpoint.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
