# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/2/29'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
# 将[0,10]等分100份
x_vals = np.linspace(0, 10, 100)
# 将[0,1]等分100份，同时将y值设置为与x相加（在x的基础上对y值进行计算）
y_vals = x_vals + np.random.normal(0, 1, 100)

# 将x变为矩阵，然后再转置，此时x_vals_column为100*1的矩阵
x_vals_column = np.transpose(np.matrix(x_vals))

# np.matrix生成矩阵，np.repeat为对第一个值的复制
'''
np.repeat(2,3)
Out[8]: array([2, 2, 2])
'''
# 将1复制100次，变为1*100的矩阵，再转置-->变为100*1的全1矩阵
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))

# 合并生成100行，2列，第一列为[0,10]的等分100份值，第二列为全1
A = np.column_stack((x_vals_column, ones_column))
# 100行1列
# b为100*1矩阵，值为[0,1)之间100个随机值
b = np.transpose(np.matrix(y_vals))

# 定义A,b为常量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# 2*100矩阵和100*2矩阵相乘，结果得到2*2矩阵
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
# 求tA_A的逆，仍为2*2矩阵
tA_A_inv = tf.matrix_inverse(tA_A)

# 2*2矩阵和2*100矩阵相乘，得到2*100矩阵
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
# 2*100矩阵和100*1矩阵相乘，得到solution为2*1矩阵
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)

slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope:', slope)
print('y_intercept:', y_intercept)

best_fit = []
# 计算最小二乘法之后的预测值
for each in x_vals:
    best_fit.append(slope * each + y_intercept)

# 做出原始数据
plt.plot(x_vals, y_vals, 'o', label="Data")
# 做出预测值

plt.plot(x_vals, best_fit, 'r-', label="Fit line", linewidth=3)
# 图例位置
plt.legend(loc='upper left')
# 显示
plt.show()
