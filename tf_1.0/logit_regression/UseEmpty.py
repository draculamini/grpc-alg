# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/30'


import tensorflow as tf

modelPath = 'model/_wx_b.pb'

with tf.Session() as sess:
    with open(modelPath, 'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        output = tf.import_graph_def(graph_def, input_map={'x:0': tf.constant(12.)}, return_elements=['out:0'])
        print(sess.run(output))

        output = tf.import_graph_def(graph_def, input_map={'x:0': tf.constant(120.)}, return_elements=['out:0'])
        print(sess.run(output))

