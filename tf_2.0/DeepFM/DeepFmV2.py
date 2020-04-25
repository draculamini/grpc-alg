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

    weight_1 = tf.keras.layers.Lambda(lambda x: x * 0.8)
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

    index = tf.cast(x, tf.int32)

    x = tf.keras.backend.random_normal(
        [100], mean=0.0, stddev=1.0, dtype=None, seed=None
    )

    print(tf.nn.embedding_lookup(x, index))


    input_dim = 39

    feat_index = tf.keras.Input(input_dim, name="feat_index", dtype=tf.int64)
    feat_value = tf.keras.Input(input_dim, name="feat_value")

    x = tf.keras.backend.random_normal(
        [100], mean=0.0, stddev=1.0, dtype=None, seed=None
    )
    fm_1_weight_table = tf.keras.backend.random_normal(
        [FEATURE_SIZE], mean=0.0, stddev=1.0, dtype=None, seed=None
    )

    fm_1_weight = tf.nn.embedding_lookup(fm_1_weight_table, feat_index)
    fm_1_factor = tf.keras.layers.multiply([fm_1_weight, feat_value])

    fm_1_model = tf.keras.Model(inputs=[feat_index, feat_value], outputs=fm_1_factor)

    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    value = initializer(shape=(2, 39))

    print(value)

    index = tf.ones(shape=[2, 39])
    index = tf.cast(index, tf.int64)

    print(fm_1_model({"feat_index": index, "feat_value": value}))

















