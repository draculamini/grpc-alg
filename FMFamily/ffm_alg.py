# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/10'


import numpy as np
import tensorflow as tf
import sys
from FMFamily.build_data import load_data


class HyperArgs(object):
    # 模型参数
    feature_size = 100  # 特征个数
    embedding_size = 256  # 嵌入维度 这里是FM第二个因子v向量维度, DNN层和FM输入相同
    deep_layers = [512, 256, 128]  # DNN各层维度数组
    field_size = 39  # 一条记录里的特征个数

    fm_field_size = 15  # 按领域区分特征的特征field个数

    # 训练中参数
    epoch = 3
    batch_size = 64
    learning_rate = 0.05
    l2_reg_rate = 0.01
    is_training = False
    checkpoint_dir = '/Users/wuxikun/Documents/gRPC/python3/FMFamily/model/'


class FFM(object):
    def __init__(self, args):

        self.embedding_size = args.embedding_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate
        self.field_size = args.field_size
        self.feature_size = args.feature_size
        self.learning_rate = args.learning_rate
        self.deep_activation = tf.nn.relu
        self.fm_field_size = args.fm_field_size

        self.weight = dict()
        self.build_model()

    def build_model(self):

        # 输入数据 feat_index 是特征下标用于 取FM算法第一个因子权重及第二因子的v向量
        self.feat_index = tf.placeholder(tf.int32, [None, None], "feat_indx")
        self.feat_value = tf.placeholder(tf.float32, [None, None], "feat_value")
        self.label = tf.placeholder(tf.float32, [None, None], "label")

        # FM算法 第一个因子
        # 第一个因子W1
        self.weight["first_actor_weight"] = tf.Variable(
            tf.random_normal([self.feature_size, 1], 0.0, 0.01), name="first_actor_weight"
        )
        # W1*X
        self.fm_first_actor = tf.multiply(tf.nn.embedding_lookup(
            self.weight["first_actor_weight"], self.feat_index
        ), tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # 求和 ∑W1i * Xi
        self.fm_first_actor = tf.reduce_sum(self.fm_first_actor, 2) # (batch_size, field_size)

        # 第二个因子
        self.weight["embedding_weight"] = tf.Variable(
            tf.random_normal([self.fm_field_size, self.feature_size, self.embedding_size], 0.0, 0.01), name="embedding_weight"
        )

        fm_result = None

        for i in range(self.fm_field_size):
            for j in range(i + 1, self.fm_field_size):

                vi_fj = tf.nn.embedding_lookup(self.weight["embedding_weight"][j], self.feat_index) # [batch, field_size, embedding_size]
                vj_fi = tf.nn.embedding_lookup(self.weight["embedding_weight"][i], self.feat_index) # [batch, field_size, embedding_size]
                wij = tf.reduce_sum(tf.multiply(vi_fj, vj_fi), axis=2) # (batch_x, field_size)

                x_i = self.feat_value[:, i] # batch
                x_j = self.feat_value[:, j] # batch
                xij = tf.multiply(x_i, x_j) # (batch )
                xij = tf.expand_dims(xij, 1)

                if (fm_result == None):
                    fm_result = tf.multiply(wij, xij)  # (batch, 1)
                else:
                    fm_result += tf.multiply(wij, xij) # (batch_x, field_size)

        merge = fm_result + self.fm_first_actor # (batch_x, field_size)

        merge_layer_size = self.field_size
        init_value = np.sqrt(np.sqrt(2.0 / (merge_layer_size + 1)))

        self.weight["merge_layer"] = tf.Variable(
            tf.random_normal([merge_layer_size, 1], 0, init_value), name="merge_layer"
        )

        self.weight["merge_bias"] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32, name="merge_bias"
        )

        self.out = tf.add(tf.matmul(merge, self.weight["merge_layer"]), self.weight["merge_bias"])


        self.out = tf.nn.sigmoid(self.out)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["merge_layer"])

        # for i in range(len(self.deep_layers)):
        #     self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["layer_%d" % i])

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])


if __name__ == '__main__':

    args = HyperArgs()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True


    with tf.Session(config=gpu_config) as sess:
        Model = FFM(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = Model.train(sess, X_index, X_value, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        Model.save(sess, args.checkpoint_dir)
        else:
            Model.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = Model.predict(sess, X_index, X_value)
                print(result)



