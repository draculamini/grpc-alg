# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/5'


import tensorflow as tf

def f(x, y):
  output = 1.0
  for i in range(y):
    if 1< i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


# x = tf.ones((2, 3))
# # print(x)
# with tf.GradientTape() as t:
#     t.watch(x)
#     y = tf.reduce_sum(x)
#     z = tf.multiply(y, y)
#
# print("y-> ", y)
# print("z-> ", z)
#
# dz_dy =  t.gradient(z, y)
#
# print("dz_dy->", dz_dy)

# dz_dx = t.gradient(z, x)
#
# for i in [0, 1]:
#     for j in [0, 1]:
#         assert dz_dx[i][j].numpy() == 8.0
#
# def f(x, y):
#     output = 1.0
#     for i in range(y):
#         if 1< x < 5:
#             output = tf.multiply(output, x)
#     return output
#
# def grad(x, y):
#   with tf.GradientTape() as t:
#     t.watch(x)
#     out = f(x, y)
#   return t.gradient(out, x)
#
# x = tf.convert_to_tensor(2.0)
#
# # assert grad(x, 6).numpy() == 12.0
# # assert grad(x, 5).numpy() == 12.0
# assert grad(x, 4).numpy() == 4.0







