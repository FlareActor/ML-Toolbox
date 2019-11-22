import argparse
import os
import pdb

import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def ckpt2pb():
    saver = tf.train.import_meta_graph('./model/tf/checkpoint.ckpt.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, './model/tf/checkpoint.ckpt')
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['fc1/add'])
        with tf.gfile.FastGFile('./model/tf/checkpoint.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def remove_layer_from_pb():
    parser = argparse.ArgumentParser(description='Remove layer from pb file')
    parser.add_argument('-pb', type=str)
    args = parser.parse_args()
    pb_path = args.pb

    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    for idx, node in enumerate(graph_def.node):
        print(idx, node.name, node.op, sep='  ')
    pdb.set_trace()

    inputs = input('Input the start and end index:\n')
    start_idx, end_idx = list(map(int, inputs.split(' ')))

    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(graph_def.node[end_idx])
    print('Old input of the end node:\n', new_node.input)
    new_input = new_node.input[:].copy()

    drop_layer_name = next(filter(lambda i: i.split('/')[0] != new_node.name.split('/')[0], new_input))
    new_input[new_input.index(drop_layer_name)] = graph_def.node[start_idx].name
    drop_layer_name = drop_layer_name.split('/')[0]
    del new_node.input[:]
    new_node.input.extend(new_input)
    print('New input of the end node:\n', new_node.input)

    new_nodes = graph_def.node[:].copy()
    new_nodes[end_idx] = new_node
    new_nodes = [n for n in new_nodes if n.name.split('/')[0] != drop_layer_name]

    output_graph_def = tf.GraphDef()
    output_graph_def.node.extend(new_nodes)
    for idx, node in enumerate(output_graph_def.node):
        print(idx, node.name, node.op, sep='  ')
    pdb.set_trace()

    with tf.gfile.GFile(os.path.splitext(pb_path)[0] + '_removed.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    # ckpt2pb()
    remove_layer_from_pb()
    pass
