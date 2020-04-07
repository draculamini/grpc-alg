# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/5'

import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


if __name__ == "__main__":
    layer = MyDenseLayer(10)

    _ = layer(tf.zeros([10, 5]))

    print([var.name for var in layer.trainable_variables])

    print(layer.kernel)

