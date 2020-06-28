# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/17'


import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Input(10,),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

if __name__ == '__main__':
    sample = np.ones(shape=[1, 10])
    print(model(sample))

    model.save("./model/denseModel.h5")