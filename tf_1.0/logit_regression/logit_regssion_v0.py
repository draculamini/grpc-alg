# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/30'


import tensorflow as tf
from numpy.random import RandomState

# 使用命名空间定义元素，便于使用tensorboard查看神经网络图形化

g1 = tf.Graph()
with g1.as_default():

    with tf.name_scope('graph_1') as scope:
        batch_size = 500  # 神经网络训练集batch大小为500
        # 定义神经网络的结构，输入为2个参数，隐藏层为10个参数，输出为1个参数
        # w1为输入到隐藏层的权重，2*10的矩阵（2表示输入层有2个因子，也就是两列输入，10表示隐藏层有10个cell）
        w1 = tf.Variable(tf.random_normal([2, 10], stddev=1, seed=1), name='w1')
        # w2为隐藏层到输出的权重，10*1的矩阵（接受隐藏的10个cell输入，输出1列数据）
        w2 = tf.Variable(tf.random_normal([10, 1], stddev=1, seed=1), name='w2')
        # b1和b2均为一行，列数对应由w1和w2的列数决定
        b1 = tf.Variable(tf.random_normal([1, 10], stddev=1, seed=1), name='b1')
        b2 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1), name='b2')

        # 维度中使用None，则可以不规定矩阵的行数，方便存储不同batch的大小。（占位符）
        x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
        y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

        # 定义神经网络前向传播的过程，定义了1层隐藏层。
        # 输入到隐藏、隐藏到输出的算法均为逻辑回归，即y=wx+b的模式
        a = tf.add(tf.matmul(x, w1, name='a'), b1)
        y = tf.add(tf.matmul(tf.tanh(a), w2, name='y'), b2)  # 使用tanh激活函数使模型非线性化
        out = tf.sigmoid(y, name="out")  # 将逻辑回归的输出概率化

        # 定义损失函数和反向传播的算法，见吴恩达视频课程第二周第三节课，逻辑回归的损失函数
        cross_entropy = - \
            tf.reduce_mean(y_ * tf.log(tf.clip_by_value(out, 1e-10, 1.0)) +
                           (1 - y_) * tf.log(tf.clip_by_value((1 - out), 1e-10, 1.0)))
        # 方差损失函数，逻辑回归不能用
        # cost = -tf.reduce_mean(tf.square(y_ - y_hat))
        # clip_by_value函数将y限制在1e-10和1.0的范围内，防止出现log0的错误，即防止梯度消失或爆发

        train_step = tf.train.AdamOptimizer(0.0001).minimize((cross_entropy))  # 反向传播算法

        # 通过随机数生成一个模拟数据集
        rdm = RandomState(1)  # rdm为伪随机数发生器，种子为1
        dataset_size = 128000
        X = rdm.rand(dataset_size, 2)  # 生成随机数，大小为128000*2的矩阵
        # x_hat = rdm.rand(1, 2)
        x_hat = []
        x_hat.append(list(X[300]))
        print(x_hat)

        # 打标签，所有x1+x2<1的都被认为是正样本，其余为负样本。
        Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]  # 列表解析格式
        print(x_hat)

        # 若x1+x2 <1为真，则int(x1+x2 <1)为1，若假，则输出为0

export_path="model/lrModel"

node = ['out']
with tf.Session() as sess:

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                g1,
                                                                node)
    # with tf.gfile.GFile(export_path, 'wb')as f:
    #     f.write(output_graph_def.SerializeToString())

    tf.train.write_graph(output_graph_def, '.', export_path, as_text=False)



# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     saver.save(sess, export_path, global_step=0, write_meta_graph=False)



# 训练
# # 创建会话
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter("logs/", sess.graph)
#     init_op = tf.global_variables_initializer()  # 所有需要初始化的值
#     sess.run(init_op)  # 初始化变量
#     print(sess.run(w1))
#     print(sess.run(w2))
#     print('x_hat =', x_hat, 'y_hat =', sess.run(y_hat, feed_dict={x: x_hat}))
#
#     '''
#     # 在训练之前神经网络权重的值,w1,w2,b1,b2的值
#     '''
#
#     # 设定训练的轮数
#     STEPS = 100000
#     for i in range(STEPS):
#         # 每次从数据集中选batch_size个数据进行训练
#         start = (i * batch_size) % dataset_size  # 训练集在数据集中的开始位置
#         # 结束位置，若超过dataset_size，则设为dataset_size
#         end = min(start + batch_size, dataset_size)
#
#         # 通过选取的样本训练神经网络并更新参数
#         sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
#         if i % 1000 == 0:
#             # 每隔一段时间计算在所有数据上的损失函数并输出
#             total_cross_entropy = sess.run(
#                 cross_entropy, feed_dict={x: X, y_: Y})
#             total_w1 = sess.run(w1)
#             total_b1 = sess.run(b1)
#             total_w2 = sess.run(w2)
#             total_b2 = sess.run(b2)
#             print("After %d training steps(s), cross entropy on all data is %g" % (
#                 i, total_cross_entropy))
#             print('w1=', total_w1, ',b1=', total_b1)
#             print('w2=', total_w2, ',b2=', total_b2)
#
#     # 在训练之后神经网络权重的值
#     print(sess.run(w1))
#     print(sess.run(w2))
#     print('x_hat =', x_hat, 'y_hat =', sess.run(y_hat, feed_dict={x: x_hat}))
