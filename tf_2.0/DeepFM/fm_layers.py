# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/26'


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow_hub as hub


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, embed_size=7, max_feature_size=10000, **kwargs):
        super(FMLayer, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.max_feature_size = max_feature_size
        self.embed_layer = tf.keras.layers.Embedding(max_feature_size, embed_size,
                                                     embeddings_initializer='uniform',
                                                     name="embedding")

        self.fm_1_weight_table = tf.keras.backend.random_normal(
            [max_feature_size], mean=0.0, stddev=1.0, dtype=None, seed=None
        )

    @tf.function
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

    def get_config(self):
        config = {
            'embed_size': self.embed_size,
            'max_feature_size': self.max_feature_size
            # 'embed_layer': self.embed_layer,
            # 'fm_1_weight_table': self.fm_1_weight_table
        }
        base_config = super(FMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    input_dim = 50
    value = initializer(shape=(2, input_dim))
    index = tf.ones(shape=[2, input_dim])

    index = tf.cast(index, tf.int64)

    input_value = tf.keras.Input(50, dtype=tf.float32, name="input_value")
    input_index = tf.keras.Input(50, dtype=tf.int32, name="input_index")
    fm_part = FMLayer(name="fm_layer")(input_value, input_index)
    out = tf.keras.layers.Dense(2)(fm_part)

    model = tf.keras.Model(inputs=[input_value, input_index], outputs=out)

    model.compile(optimizer=u'adam', loss="binary_crossentropy")

    print("model \n ", model({"input_value": value, "input_index": index}))
    print("model \n ", model({"input_value": value, "input_index": index}))

    print(model.get_config())

    model_path = "../model/fmModel.h5"
    model.save(model_path)
    weight_path = "../weight/fm_model"

    model.save_weights(weight_path)

    # new_model = tf.keras.models.load_model(model_path, custom_objects={'FMLayer': FMLayer})
    new_model = tf.keras.models.load_model(model_path, custom_objects={'FMLayer': FMLayer})
    # new_model.load_weights(weight_path)

    print("new_model \n ", new_model({"input_value": value, "input_index": index}))
    # print(new_model.get_config())

    # print(model.get_weights())  # Retrieves the state of the model.
    # print(new_model.get_weights())  # Retrieves the state of the model.

    fm_layer_weights = model.get_layer('fm_layer').get_weights()
    print(fm_layer_weights)


