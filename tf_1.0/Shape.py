# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/6/9'


import tensorflow as tf
a=tf.constant([[1,2],
               [3,4],
               [5,6]])
with tf.Session() as sess:
    print(a.shape)
    print(tf.shape(a))
    print(sess.run(tf.shape(a)))
