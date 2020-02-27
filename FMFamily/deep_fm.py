# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/2/25'


import numpy as np
import tensorflow as tf
import sys
from FMFamily.build_data import load_data


class HyperArg():

    # 模型参数
    feature_size = 100
    field_size = 15
    embedding_size = 256
    deep_layers = [512, 256, 128]


    # 训练参数
    epoch = 3
    batch_size = 64
    lr_rate = 1.0
    l2_reg_rate = 0.01
    is_training = True
    checkpoint_dir = '/Users/wuxikun/Documents/gRPC/python3/FMFamily/model/'


class DeepFM():

    def __init__(self, arg):

        self.embedding_size = arg.embedding_size
        self.deep_layers = [512, 256, 128]
        self.field_size = arg.field_size
        self.feature_size = arg.feature_size
        self.weight = dict()
        self.deep_activation = tf.nn.relu
        self.learning_rate = arg.lr_rate
        self.l2_reg_rate = args.l2_reg_rate

        self.epoch = args.epoch
        self.build_model()


    def build_model(self):

        # 输入数据
        self.feat_index = tf.placeholder(tf.int32, [None, None], "feature_index")
        self.feat_value = tf.placeholder(tf.float32, [None, None], "feature_value")
        self.label = tf.placeholder(tf.float32, [None, None], "label")

        # 1.embeding层参数 FM公式中的V

        self.weight["embedding_weight"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="embedding_weight"
        )

        # 2.1 FM 一次项(∑WX) 的参数
        self.weight["fm_first_weight"] = tf.Variable(
            tf.random_normal([self.feature_size, 1], 0.0, 0.01),
            name="fm_first_weight"
        )

        # 2.2 FM 二次项参数

        # 3. deep层各层参数
        cnt_of_deep_layers = len(self.deep_layers)
        deep_input_size = self.field_size * self.embedding_size
        init_value = np.sqrt(2.0 / (deep_input_size + self.deep_layers[0]))

        self.weight["layer_0"] = tf.Variable(
            tf.random_normal([deep_input_size, self.deep_layers[0]], 0.0, init_value), dtype=tf.float32, name="layer_0"
        )

        self.weight["bias_0"] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0.0, init_value), dtype=tf.float32, name="bias_0"
        )
        # 其它层用的参数
        if cnt_of_deep_layers > 1:
            for i in range(1, cnt_of_deep_layers):
                init_method = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weight["layer_" + str(i)] = tf.Variable(
                    tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0.0, init_method), dtype=tf.float32,
                    name="layer_" + str(i)
                )
                self.weight["bias_" + str(i)] = tf.Variable(
                    tf.random_normal([1, self.deep_layers[i]], 0.0, init_value), dtype=tf.float32, name="bias_" + str(i)
                )

        # 最后一层 合并
        merge_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size

        init_method = np.sqrt(np.sqrt(2.0 / (merge_layer_size + 1)))

        self.weight["merge_layer"] = tf.Variable(
            tf.random_normal([merge_layer_size, 1], 0, init_method), name="merge_layer"
        )
        self.weight["merge_bias"] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32, name="merge_bias"
        )

        # 嵌入层
        self.embedding_index = tf.nn.embedding_lookup(self.weight["embedding_weight"], self.feat_index)

        self.embedding_part = tf.multiply(self.embedding_index, tf.reshape(
            self.feat_value, [-1, self.field_size, 1]
        ))

        # self.weight['fm_first_weight'] batch * feature_size * 1
        self.fm_first_factor = tf.nn.embedding_lookup(self.weight['fm_first_weight'], self.feat_index)

        self.fm_first_factor = tf.multiply(self.fm_first_factor, tf.reshape(self.feat_value, [-1, self.field_size, 1]))

        self.fm_first_factor = tf.reduce_sum(self.fm_first_factor, 2)  # (batch_size, feat_size, 1)

        self.sum_second_factor = tf.reduce_sum(self.embedding_part, 1)
        self.sum_second_factor_square = tf.square(self.sum_second_factor)

        self.square_second_factor = tf.square(self.embedding_part)
        self.square_second_factor_sum = tf.reduce_sum(self.square_second_factor, 1)

        self.fm_second_factor = 0.1 * (tf.subtract(self.sum_second_factor_square, self.square_second_factor_sum))

        self.fm_part = tf.concat([self.fm_first_factor, self.fm_second_factor], axis=1)

        self.deep_embedding = tf.reshape(self.embedding_part, [-1, self.field_size * self.embedding_size])

        for i in range(0, len(self.deep_layers)):
            self.deep_embedding = self.deep_activation(tf.add(tf.matmul(self.deep_embedding, self.weight["layer_" + str(i)]),
                                         self.weight["bias_" + str(i)]))

        concat_all = tf.concat([self.fm_part, self.deep_embedding], axis=1)
        self.out = tf.add(tf.matmul(concat_all, self.weight["merge_layer"]), self.weight["merge_bias"])

        self.out = tf.nn.sigmoid(self.out)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1-self.label) * tf.log(1 - self.out + 1e-24))

        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["merge_layer"])

        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["layer_%d" % i])

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
    args = HyperArg()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session(config=gpu_config) as sess:
        Model = DeepFM(args)
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


























