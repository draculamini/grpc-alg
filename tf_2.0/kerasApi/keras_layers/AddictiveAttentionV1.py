# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/17'


import tensorflow as tf
import numpy as np

max_tokens = 100000
dimension = 256

query_input = tf.keras.Input(shape=(None,), dtype='int32')
value_input = tf.keras.Input(shape=(None,), dtype='int32')

# token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
#
# query_embedding = token_embedding(query_input)
# value_embedding = token_embedding(query_input)

cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    padding='same'
)

# query_seq_encoding = cnn_layer(query_embedding)
# value_seq_encoding = cnn_layer(value_embedding)

import tensorflow.keras.backend as K


def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

if __name__ == '__main__':

    # sample = np.ones(shape=[64, 200, 256])
    # sample = np.ones(shape=[64, 200, 256])
    # x = cnn_layer(sample)
    # print(x.shape)

    sample = np.ones(shape=[64, 200, 256])
    x = cnn_layer(sample)

    query_value_attention_seq = tf.keras.layers.Attention()(
        [x, x])

    print(query_value_attention_seq.shape)




















