# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/6/12'

import tensorflow as tf
import numpy as np
LAYER_1_NAME = 'layer1'  # 第一层的名字
LAYER_2_NAME = 'layer2'  # 第二层的名字

# 创建一个非常简单的神经网络，它有两层
x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
layer1 = tf.layers.dense(x, 5, activation=tf.nn.sigmoid, name=LAYER_1_NAME)
layer2 = tf.layers.dense(layer1, 2, activation=tf.nn.sigmoid, name=LAYER_2_NAME)
loss = tf.reduce_mean((layer2 - x) ** 2)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_values = np.random.normal(0, 1, (5000, 2))  # 生成用于输入的随机数
    for step in range(1000):
        _, loss_value = sess.run([optimizer, loss], feed_dict={x: x_values})
        if step % 100 == 0:
            print("step: %d, loss: %f" % (step, loss_value))
    # 把模型保存成    checkpoint
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, './checkpoint/model.ckpt')
    print("model saved in path: %s" % save_path, flush=True)
# 读取刚保存的checkpoint
reader = tf.train.NewCheckpointReader(save_path)
weights = reader.get_tensor(LAYER_1_NAME + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
bias = reader.get_tensor(LAYER_1_NAME + '/bias')  # bias的名字
print(weights)
print(bias)
# 如果想打印模型中的所有参数名和参数值的话，把下面几行取消注释
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor name: ", key)
    #     print(reader.get_tensor(key))