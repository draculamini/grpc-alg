# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/15'

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import functools


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

print(train_file_path)
print(test_file_path)

np.set_printoptions(precision=3, suppress=True)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
LABEL_COLUMN = 'survived'


def get_dataset(file_path):

    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=5,
        ignore_errors=True)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_data))

print(examples)
# print(labels)

CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}

categorical_columns = []

for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab
    )

    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

print(categorical_columns)


def process_continuous_data(mean, data):

    # 标准化数据
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])


MEANS = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)
print(numerical_columns)

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

tf.print("preprocessing_layer(examples) \n ", preprocessing_layer(examples))

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


train_data = raw_train_data.shuffle(500)
test_data = raw_test_data
model.fit(train_data, epochs=1)
tf.print(model.summary())

test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))








