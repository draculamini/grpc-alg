# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/24'


import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import numpy as np


class DeepFM(tf.keras.Model):

    def __init__(self):

        super(DeepFM, self).__init__


    def call(self, inputs):
        pass

if __name__ == '__main__':
    kvar = tf.keras.backend.eye(3)
    print(kvar)

    FEATURE_SIZE=10000
    EMBEDDIM=128

    # feat_index = tf.keras.Input(None, name="feat_index")
    # feat_value = tf.keras.Input(None, name="feat_value")
    # weight = K.variable(np.ones((FEATURE_SIZE,), dtype='float32') * 0.8)

    weight_1 = tf.keras.layers.Lambda(lambda x: x * 0.8)

    # weight = tf.keras.layers.Embedding(shape=(feat_value.shape[1], 1))
    # fm_first_factor = tf.keras.layers.Multiply(weight, feat_value)

    # embed = tf.keras.layers.Embedding(shape=(FEATURE_SIZE, EMBEDDIM))(feat_index)

    # model = tf.keras.Model(feat_value, fm_first_factor)

    # x = tf.tr([20])
    # y = tf.random([20])
    # print(x)
    #
    # sum = tf.keras.layers.multiply([x, y])
    # print(sum)

    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    x = initializer(shape=(1, 20))

    y = initializer(shape=(1, 20))
    print(x)
    print(y)
    print(tf.keras.layers.multiply([x, y]))

    x = tf.linspace(1., 10., 10, name=None)
    y = tf.linspace(1., 10., 10, name=None)

    print(x)
    print(y)
    print(tf.keras.layers.multiply([x, y]))

    x = tf.keras.backend.random_normal(
        [10], mean=0.0, stddev=1.0, dtype=None, seed=None
    )
    print(x)













