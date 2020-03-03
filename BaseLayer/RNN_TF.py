# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/2'


import tensorflow as tf
import BaseLayer.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class MnistRNN():

    def __init__(self):
        # 先定义几个后面要用到的常量
        # 时间节点
        self.time_steps = 28
        # LSTM 隐藏层单元数量
        self.num_units = 128
        # 一行输入28个像素点
        self.n_input = 28
        # adam的学习率
        self.learning_rate = 0.001
        # 标签种类 (0-9).
        self.n_classes = 10



        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.float32, [None, 784], name="x")

        self.y = tf.placeholder(tf.float32, [None, 10], name="y")

        image = tf.reshape(self.x, [-1, self.time_steps, self.n_input])

        # weights & bias
        out_weights = tf.Variable(tf.random_normal([self.num_units, self.n_classes]))
        out_bias = tf.Variable(tf.random_normal([self.n_classes]))

        input = tf.unstack(self.x, self.time_steps, 1)

        # 定义layer和rnn训练
        lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=1)
        outputs, _ = tf.nn.static_rnn(lstm_layer, input, dtype="float32")

        prediction = tf.matmul(outputs[-1], out_weights) + out_bias

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # 计算准确度
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))