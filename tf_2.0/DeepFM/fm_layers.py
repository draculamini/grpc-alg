# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/26'


import tensorflow as tf
from tensorflow.keras import backend as K


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size=7, max_feature_size=10000):
        super(FMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.max_feature_size = max_feature_size
        self.embed_layer = tf.keras.layers.Embedding(max_feature_size, hidden_size,
                                                     embeddings_initializer='uniform',
                                                     name="embedding")

        self.fm_1_weight_table = tf.keras.backend.random_normal(
            [max_feature_size], mean=0.0, stddev=1.0, dtype=None, seed=None
        )

    def call(self, feat_value, feat_index):
        fm_1_weight = tf.nn.embedding_lookup(self.fm_1_weight_table, feat_index)
        fm_1_factor = tf.keras.layers.multiply([fm_1_weight, feat_value])
        embed = self.embed_layer(feat_index)
        tmp = tf.reshape(feat_value, [-1, K.shape(feat_value)[-1], 1])
        embed_part = tf.keras.layers.multiply([embed, tmp])

        second_factor_sum = tf.math.reduce_sum(embed_part, 1)
        second_factor_sum_square = tf.math.square(second_factor_sum)
        second_factor_square = tf.math.square(embed_part)
        second_factor_square_sum = tf.math.reduce_sum(second_factor_square, 1)
        fm_2_factor = 0.5 * tf.math.subtract(second_factor_sum_square, second_factor_square_sum)
        return tf.keras.layers.concatenate([fm_1_factor, fm_2_factor], axis=-1)


if __name__ == '__main__':
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    input_dim = 50
    value = initializer(shape=(2, input_dim))
    index = tf.ones(shape=[2, input_dim])
    index = tf.cast(index, tf.int64)
    layer = FMLayer()
    print(layer(value, index))
    input_value = tf.keras.Input(50, dtype=tf.float32, name="input_value")
    input_index = tf.keras.Input(50, dtype=tf.int32, name="input_index")
    fm_part = FMLayer()(input_value, input_index)
    out = tf.keras.layers.Dense(2)(fm_part)
    model = tf.keras.Model(inputs=[input_value, input_index], outputs=out)

    print(model({"input_value": value, "input_index": index}))

