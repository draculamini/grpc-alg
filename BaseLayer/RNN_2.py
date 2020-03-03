# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/2'


import tensorflow as tf
from tensorflow.contrib import rnn
import BaseLayer.input_data as input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 先定义几个后面要用到的常量
# 时间节点
time_steps = 28
# LSTM 隐藏层单元数量
num_units = 128
# 一行输入28个像素点
n_input = 28
# adam的学习率
learning_rate = 0.001
# 标签种类 (0-9).
n_classes = 10
# 一个批次（batch）的数量
batch_size = 128

# weights & bias
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# 两个占位符
x = tf.placeholder("float", [None, time_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 上面x的格式需要转换为[batch_size,n_input]才能作为input 创给static_rnn方法
# 转化以后 input就变成格式为 [batch_size,n_input] 长度为time_teps 也就是28的一个list
# 我们将这个list也称为tensor
input = tf.unstack(x, time_steps, 1)

# 定义layer和rnn训练
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# outputs (n_input, ?, num_units)
# 只考虑最后时间节点的输出 作为我们的预测值
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

# 优化器
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 计算准确度
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 800:
        # 获取下一批次的数据
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        # 改变x的结构，获取到的数据格式为[number,784] 要改变才能匹配我们定义的placehodler
        batch_x = batch_x.reshape((batch_size, time_steps, n_input))
        # 运行
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter % 10 == 0:
            # 计算训练集的准确度
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

        iter = iter + 1

    # 也可以计算测试集的准确度 参考结果 ： 99.21%
    test_data = mnist.test.images.reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

