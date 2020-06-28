# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/16'


import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)

print(df.head())
print(df.dtypes)

print(df['thal'].head())

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print(df.head())

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

print(tf.constant(df['thal']))


train_dataset = dataset.shuffle(len(df)).batch(1)


def get_compiled_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

# model = get_compiled_model()
# model.fit(train_dataset, epochs=15)



