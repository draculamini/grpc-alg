# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/3'

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import keras

import tensorflow as tf


class HyperArgs():
    embedding_dim = 128
    max_feature = 1000
    max_len = 200
    num_classes = 2
    learning_rate = 0.01

    EPOCH = 20
    batch_size = 128


class SelfDotAtt():

    def __init__(self, args):

        self.embedding_dim = args.embedding_dim
        self.max_feature = args.max_feature
        self.max_len = args.max_len
        self.num_classes = args.num_classes
        self.learning_rate = args.learning_rate

        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.placeholder(tf.float32, [None, 2])
        self.embedding_dict = tf.Variable(
            tf.random_normal([self.max_feature, self.embedding_dim], 0, 0.01),
            name="embedding_dict"
        )
        embed = tf.nn.embedding_lookup(self.embedding_dict, self.x)  # (batch_size, maxlen, embeding_dim)
        # self attention y = softmax( Q * K / sqrt(k)) * V
        Q, K, V = embed, embed, embed
        Q_K = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        Q_K_V = tf.matmul(tf.div(tf.nn.softmax(Q_K), tf.sqrt(self.embedding_dim * 1.0)), V)

        fc = tf.reshape(Q_K_V, [-1, self.embedding_dim * self.max_len])

        out_weight = tf.Variable(
            tf.random_normal([self.embedding_dim * self.max_len, self.num_classes]),
            name="out_weight"
        )

        self.out = tf.matmul(fc, out_weight)
        self.out = tf.sigmoid(self.out)

        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.out))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def train(self, sess, batch_x, batch_y):

        _, accuracy = sess.run([self.train_step, self.accuracy], feed_dict={
            self.x: batch_x,
            self.y: batch_y
        })

        return accuracy



if __name__ == '__main__':

    args = HyperArgs
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=args.max_feature, maxlen=args.max_len)

    #  数据预处理
    train_x = pad_sequences(train_x, maxlen=args.max_len)
    test_x = pad_sequences(test_x, maxlen=args.max_len)

    train_y = keras.utils.np_utils.to_categorical(train_y, 2)
    test_y = keras.utils.np_utils.to_categorical(test_y, 2)

    selfDotModel = SelfDotAtt(args)

    for i in range(args.EPOCH):

        current_index = 0
        max_size = train_x.shape[0]

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            while current_index + args.batch_size < max_size:

                batch_x = train_x[current_index: current_index + args.batch_size]
                batch_y = train_y[current_index: current_index + args.batch_size]

                acc = selfDotModel.train(sess, batch_x, batch_y)
                current_index += args.batch_size
                # print("Epoch : ", i, " current_index: ", current_index, "max_size", max_size , "acc :", acc)

            acc = selfDotModel.train(sess, test_x, test_y)
            print("Epoch : ", i,  "test acc :", acc)









