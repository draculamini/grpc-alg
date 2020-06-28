# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/24'

import tensorflow as tf
# # 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# # 注意我们可以通过传递一个 `name` 参数来命名任何层。
# main_input = tf.keras.Input(shape=(100,), dtype='int32', name='main_input')
#
# # Embedding 层将输入序列编码为一个稠密向量的序列，每个向量维度为 512。
# x = tf.keras.layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#
# # LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
# lstm_out = tf.keras.layers.LSTM(32)(x)
#
# auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#
# auxiliary_input = tf.keras.Input(shape=(5,), name='aux_input')
# x = tf.keras.layers.concatenate([lstm_out, auxiliary_input])
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)
#
# model = tf.keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',
#               loss_weights=[1., 0.2])

import tensorflow as tf


raw_tensor = tf.ones([2, 50, 1])
print(raw_tensor)


squeezed_tensor = tf.squeeze(input=raw_tensor, axis=[-1])
print(squeezed_tensor)




# # -*- coding:utf-8 -*-
# # @version: 1.0
# # @author: wuxikun
# # @date: '2020/4/24'
#
#
# import tensorflow as tf
#
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import *
# import numpy as np
#
#
# class DeepFM(tf.keras.Model):
#
#     def __init__(self):
#
#         super(DeepFM, self).__init__
#
#
#     def call(self, inputs):
#         pass
#
# if __name__ == '__main__':
#     kvar = tf.keras.backend.eye(3)
#     print(kvar)
#
#     FEATURE_SIZE=10000
#     EMBEDDIM=128
#
#     # feat_index = tf.keras.Input(None, name="feat_index")
#     # feat_value = tf.keras.Input(None, name="feat_value")
#     # weight = K.variable(np.ones((FEATURE_SIZE,), dtype='float32') * 0.8)
#
#     weight_1 = tf.keras.layers.Lambda(lambda x: x * 0.8)
#
#     # weight = tf.keras.layers.Embedding(shape=(feat_value.shape[1], 1))
#     # fm_first_factor = tf.keras.layers.Multiply(weight, feat_value)
#
#     # embed = tf.keras.layers.Embedding(shape=(FEATURE_SIZE, EMBEDDIM))(feat_index)
#
#     # model = tf.keras.Model(feat_value, fm_first_factor)
#
#     # x = tf.tr([20])
#     # y = tf.random([20])
#     # print(x)
#     #
#     # sum = tf.keras.layers.multiply([x, y])
#     # print(sum)
#
#     initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
#     x = initializer(shape=(1, 20))
#
#     y = initializer(shape=(1, 20))
#     print(x)
#     print(y)
#     print(tf.keras.layers.multiply([x, y]))
#
#     x = tf.linspace(1., 10., 10, name=None)
#     y = tf.linspace(1., 10., 10, name=None)
#
#     print(x)
#     print(y)
#     print(tf.keras.layers.multiply([x, y]))
#
#     index = tf.cast(x, tf.int32)
#
#     x = tf.keras.backend.random_normal(
#         [100], mean=0.0, stddev=1.0, dtype=None, seed=None
#     )
#
#
#     # index = tf.linspace(1, 10, 12)
#     # print(index)
#     print(tf.nn.embedding_lookup(x, index))
#     # print(x)













