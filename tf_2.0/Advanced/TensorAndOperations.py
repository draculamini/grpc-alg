# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/4'


import tensorflow as tf
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1,2, 3]))
print(tf.square(2) + tf.square(3))

x = tf.matmul([[2]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

import numpy as np

ndarray = np.ones([3, 3])
tensor = tf.multiply(ndarray, 42)
print(tensor)

print(np.add(tensor, 1))
print(tensor.numpy())
