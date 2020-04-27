# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/26'


import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_hub as hub


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, embed_size=7, max_feature_size=10000, **kwargs):
        super(FMLayer, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.max_feature_size = max_feature_size
        self.embed_layer = tf.keras.layers.Embedding(max_feature_size, embed_size,
                                                     embeddings_initializer='uniform',
                                                     name="embedding")

        self.fm_1_weight_table = tf.keras.layers.Embedding(max_feature_size, 1,
                                                     embeddings_initializer='uniform',
                                                     name="fm_1_weight_table")

    @tf.function
    def call(self, feat_value, feat_index):

        fm_1_weight = tf.squeeze(self.fm_1_weight_table(feat_index), axis=[-1])
        fm_1_factor = tf.keras.layers.multiply([fm_1_weight, feat_value])

        embed = self.embed_layer(feat_index)
        tmp = tf.reshape(feat_value, [-1, K.shape(feat_value)[-1], 1])
        embed_part = tf.keras.layers.multiply([embed, tmp])

        second_factor_sum = tf.math.reduce_sum(embed_part, 1)
        second_factor_sum_square = tf.math.square(second_factor_sum)
        second_factor_square = tf.math.square(embed_part)
        second_factor_square_sum = tf.math.reduce_sum(second_factor_square, 1)
        fm_2_factor = 0.5 * tf.math.subtract(second_factor_sum_square, second_factor_square_sum)

        return tf.concat([fm_1_factor, fm_2_factor], axis=1), embed

    def get_config(self):
        config = {
            'embed_size': self.embed_size,
            'max_feature_size': self.max_feature_size

        }
        base_config = super(FMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    input_dim = 50

    value = initializer(shape=(2, input_dim))
    index = tf.ones(shape=[2, input_dim])
    index = tf.cast(index, tf.int64)

    # FM Model
    # input_value = tf.keras.Input(input_dim, dtype=tf.float32, name="input_value")
    # input_index = tf.keras.Input(input_dim, dtype=tf.int32, name="input_index")
    # fm_part, _ = FMLayer(name="fm_layer")(input_value, input_index)
    # out = tf.keras.layers.Dense(2)(fm_part)
    # model = tf.keras.Model(inputs=[input_value, input_index], outputs=out)
    # model.compile(optimizer=u'adam', loss="binary_crossentropy")
    # print("model \n ", model({"input_value": value, "input_index": index}))

    # DEEP FM Model
    input_value = tf.keras.Input(input_dim, dtype=tf.float32, name="input_value")
    input_index = tf.keras.Input(input_dim, dtype=tf.int32, name="input_index")
    fm_part, embed = FMLayer(name="fm_layer")(input_value, input_index)
    # deep Part
    flat = tf.keras.layers.Flatten()(embed)
    dnn1 = tf.keras.layers.Dense(128)(flat)
    dnn2 = tf.keras.layers.Dense(256)(dnn1)
    deepFm = tf.keras.layers.concatenate([fm_part, dnn2], axis=1)
    out = tf.keras.layers.Dense(2)(deepFm)
    model = tf.keras.Model(inputs=[input_value, input_index], outputs=out)
    print("model \n ", model({"input_value": value, "input_index": index}))

    model_path = "../model/fmModel.h5"
    model.save(model_path)
    weight_path = "../weight/fm_model"
    model.save_weights(weight_path)
    new_model = tf.keras.models.load_model(model_path, custom_objects={'FMLayer': FMLayer})
    print("new_model \n ", new_model({"input_value": value, "input_index": index}))



