# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/2'

import tensorflow as tf

# b = tf.constant([[1, 2, 3], [4, 5, 6]])
# print("b.shape:", b.shape)
# print("b:", b)
# print("-" * 50)
# b2 = b[:, tf.newaxis]
# print("b2.shape:", b2.shape)
# print("b2:", b2)

pets = {'pets': ['rabbit','pig','dog','mouse','cat']}

column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat','dog','rabbit','pig'],
    dtype=tf.string,
    default_value=-1,
    num_oov_buckets=3)

indicator = tf.feature_column.indicator_column(column)
# tensor = tf.feature_column.in(pets, [indicator])
tensor = tf.feature_column.input_layer(pets, [indicator])
print(indicator)
print(tensor)

# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     session.run(tf.tables_initializer())
#     print(session.run([indicator]))
