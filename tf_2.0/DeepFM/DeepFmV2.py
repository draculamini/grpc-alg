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
    V_SIZE=5

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

    # x = tf.keras.backend.random_normal(
    #     [100], mean=0.0, stddev=1.0, dtype=None, seed=None
    # )
    fm_1_weight_table = tf.keras.layers.Embedding(
        FEATURE_SIZE, 1, embeddings_initializer='uniform', name="embedding"
    )
    fm_1_weight = fm_1_weight_table(feat_index)
    # fm_1_weight (2, 39, 1)
    fm_1_weight = tf.squeeze(fm_1_weight)
    # feat_value (2, 39)
    # fm_1_factor = tf.keras.layers.multiply([fm_1_weight, feat_value])
    fm_1_factor = tf.multiply(fm_1_weight, feat_value)

    embed = tf.keras.layers.Embedding(FEATURE_SIZE, V_SIZE, embeddings_initializer='uniform', name="embedding")(feat_index)

    tmp = tf.reshape(feat_value, [-1, K.shape(feat_value)[-1], 1])
    embed_part = tf.keras.layers.multiply([embed, tmp])

    second_factor_sum = tf.math.reduce_sum(embed_part, 1)
    second_factor_sum_square = tf.math.square(second_factor_sum)

    second_factor_square = tf.math.square(embed_part)
    second_factor_square_sum = tf.math.reduce_sum(second_factor_square, 1)

    second_factor = 0.5 * tf.math.subtract(second_factor_sum_square , second_factor_square_sum)

    fm_1_model = tf.keras.Model(inputs=[feat_index, feat_value], outputs=second_factor)

    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    value = initializer(shape=(2, 39))

    print(value)

    index = tf.ones(shape=[2, 39])
    index = tf.cast(index, tf.int64)

    print(fm_1_model({"feat_index": index, "feat_value": value}))

















