# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/6/9'

import tensorflow as tf

x = tf.constant([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])

with tf.Session() as sess:
    print (sess.run(tf.math.reduce_euclidean_norm(x)))


