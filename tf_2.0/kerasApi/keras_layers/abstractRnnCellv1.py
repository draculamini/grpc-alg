# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/16'


import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import AbstractRNNCell

class MinimalRNNCell(AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer='uniform',
                                                name='recurrent_kernel')
        self.built=True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, output

if __name__ == '__main__':
    # model = MinimalRNNCell(128)

    input1 = tf.keras.layers.Input(shape=(16,))
    x1 = tf.keras.layers.Dense(8, activation='relu')(input1)

    input2 = tf.keras.layers.Input(shape=(32,))
    x2 = tf.keras.layers.Dense(8, activation='relu')(input2)

    input3 = tf.keras.layers.Input(shape=(64,))
    x3 = tf.keras.layers.Dense(8, activation='relu')(input3)

    added = tf.keras.layers.Add()([x1, x2, x3])
    out = tf.keras.layers.Dense(4)(added)
    model = tf.keras.models.Model(inputs=[input1, input2, input3], outputs=added)

    import numpy as np
    s1 = np.ones(shape=[1, 16])
    s2 = np.ones(shape=[1, 32])
    s3 = np.ones(shape=[1, 64])
    print(model([s1, s2, s3]))










