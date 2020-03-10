# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/3'

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf


class HyperArgs():
    embedding_dim = 128
    max_feature = 200
    max_len = 200
    num_classes = 2
    learning_rate = 0.001

    num_head = 8
    size_per_head = 32

    EPOCH = 100
    batch_size = 64


class SelfDotAtt():

    def __init__(self, args):

        self.embedding_dim = args.embedding_dim
        self.max_feature = args.max_feature
        self.max_len = args.max_len
        self.num_classes = args.num_classes
        self.learning_rate = args.learning_rate

        self.num_head = args.num_head
        self.size_per_head = args.size_per_head
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.placeholder(tf.float32, [None])

        out_put_dim = self.num_head * self.size_per_head

        self.label = tf.reshape(self.y, [-1, 1])

        self.embedding_dict = tf.Variable(
            tf.random_normal([self.max_feature, self.embedding_dim], 0, 0.01),
            name="embedding_dict", trainable=True
        )
        embed = tf.nn.embedding_lookup(self.embedding_dict, self.x)  # (batch_size, maxlen, embeding_dim)
        # self attention y = softmax( Q * K / sqrt(k)) * V
        Q, K, V = embed, embed, embed
        Q_K = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        soft_max_QK = tf.nn.softmax(tf.div(tf.nn.softmax(Q_K), tf.sqrt(self.max_len * 1.0)))
        soft_max_QK = tf.nn.dropout(soft_max_QK, 0.2)
        Q_K_V = tf.matmul(soft_max_QK, V)
        fc = tf.reshape(Q_K_V, [-1, self.embedding_dim * self.max_len])
        fc = tf.nn.dropout(fc, 0.5)
        out_weight = tf.Variable(
            tf.random_normal([self.embedding_dim * self.max_len, 1]),
            name="out_weight"
        )

        self.out = tf.matmul(fc, out_weight)

        self.out = tf.sigmoid(self.out)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        one = tf.ones_like(self.out)
        zero = tf.zeros_like(self.out)
        tmp = tf.where(self.out < 0.5, x=zero, y=one)
        self.correct_prediction = tf.equal(tf.cast(tmp, "float"), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def train(self, sess, batch_x, batch_y):

        _, accuracy, loss = sess.run([self.train_step, self.accuracy, self.loss], feed_dict={
            self.x: batch_x,
            self.y: batch_y
        })

        return accuracy, loss


class MultiHeadAtt():

    def __init__(self, args):
        self.embedding_dim = args.embedding_dim
        self.max_feature = args.max_feature
        self.max_len = args.max_len
        self.num_classes = args.num_classes
        self.learning_rate = args.learning_rate
        self.num_head = args.num_head
        self.size_per_head = args.size_per_head
        self.out_dim = self.num_head * self.size_per_head

        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.placeholder(tf.float32, [None])

        self.label = tf.reshape(self.y, [-1, 1])

        self.embedding_dict = tf.Variable(
            tf.random_normal([self.max_feature, self.embedding_dim], 0, 0.01),
            name="embedding_dict", trainable=True
        )


        embed = tf.nn.embedding_lookup(self.embedding_dict, self.x)  # (batch_size, maxlen, embeding_dim)
        # self attention y = softmax( Q * K / sqrt(k)) * V
        self.weight_Q = tf.Variable(
            tf.random_normal([self.embedding_dim, self.out_dim], 0, 0.01),
            name="weight_Q", trainable=True
        )

        self.weight_K = tf.Variable(
            tf.random_normal([self.embedding_dim, self.out_dim], 0, 0.01),
            name="weight_K", trainable=True
        )

        self.weight_V = tf.Variable(
            tf.random_normal([self.embedding_dim, self.out_dim], 0, 0.01),
            name="weight_V", trainable=True
        )

        batch_size = tf.shape(embed)[0]

        weight_q = tf.tile(self.weight_Q, [batch_size, 1])
        weight_k = tf.tile(self.weight_K, [batch_size, 1])
        weight_v = tf.tile(self.weight_V, [batch_size, 1])

        weight_q = tf.reshape(weight_q, [-1, self.embedding_dim, self.out_dim])
        weight_k = tf.reshape(weight_k, [-1, self.embedding_dim, self.out_dim])
        weight_v = tf.reshape(weight_v, [-1, self.embedding_dim, self.out_dim])

        Q = tf.matmul(embed, weight_q)
        K = tf.matmul(embed, weight_k)
        V = tf.matmul(embed, weight_v)

        Q = tf.reshape(Q, [-1, self.max_len, self.num_head, self.size_per_head])
        K = tf.reshape(K, [-1, self.max_len, self.num_head, self.size_per_head])
        V = tf.reshape(V, [-1, self.max_len, self.num_head, self.size_per_head])


        Q_K = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))
        soft_max_QK = tf.nn.softmax(tf.div(tf.nn.softmax(Q_K), tf.sqrt(self.max_len * 1.0)))
        soft_max_QK = tf.nn.dropout(soft_max_QK, 0.2)
        Q_K_V = tf.matmul(soft_max_QK, V)
        fc = tf.reshape(Q_K_V, [-1, self.out_dim * self.max_len])
        fc = tf.nn.dropout(fc, 0.5)
        out_weight = tf.Variable(
            tf.random_normal([self.out_dim * self.max_len, 1]),
            name="out_weight"
        )

        self.out = tf.matmul(fc, out_weight)

        self.out = tf.sigmoid(self.out)

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        one = tf.ones_like(self.out)
        zero = tf.zeros_like(self.out)
        tmp = tf.where(self.out < 0.5, x=zero, y=one)
        self.correct_prediction = tf.equal(tf.cast(tmp, "float"), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def train(self, sess, batch_x, batch_y):
        _, accuracy, loss = sess.run([self.train_step, self.accuracy, self.loss], feed_dict={
            self.x: batch_x,
            self.y: batch_y
        })

        return accuracy, loss


if __name__ == '__main__':

    args = HyperArgs
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=args.max_feature, maxlen=args.max_len)

    #  数据预处理
    train_x = pad_sequences(train_x, maxlen=args.max_len)
    test_x = pad_sequences(test_x, maxlen=args.max_len)

    # selfDotModel = SelfDotAtt(args)
    selfDotModel = MultiHeadAtt(args)

    for i in range(args.EPOCH):

        current_index = 0
        max_size = train_x.shape[0]

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)

            while current_index + args.batch_size < max_size:

                batch_x = train_x[current_index: current_index + args.batch_size]
                batch_y = train_y[current_index: current_index + args.batch_size]

                acc, loss = selfDotModel.train(sess, batch_x, batch_y)

                current_index += args.batch_size
                print("Epoch : ", i, " current_index: ", current_index, "max_size", max_size, "acc :", acc, "loss: ", loss)

            acc, loss = selfDotModel.train(sess, test_x, test_y)
            print("Epoch : ", i,  "test acc :", acc, " test loss :", loss)









