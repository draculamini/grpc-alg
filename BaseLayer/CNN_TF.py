# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/3/2'

import tensorflow as tf

import BaseLayer.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class MnistCNN():

    def __init__(self):
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.float32, [None, 784], name="x")
        self.y = tf.placeholder(tf.float32, [None, 10], name="y")

        image = tf.reshape(self.x, [-1, 28, 28, 1])
        filter_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], 0.0, 0.1))
        biase_1 = tf.Variable(tf.constant(0.1, shape=[32]))

        conv1 = tf.nn.conv2d(image, filter=filter_1, strides=[1, 1, 1, 1], padding="SAME")  # (batch_size, 28, 28, 32)

        h_conv1 = tf.nn.relu(conv1 + biase_1)

        pool_layer = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")   # (batch_size, 14, 14, 32)

        fc = tf.reshape(pool_layer, [-1, 14*14*32])

        out_weight = tf.Variable(tf.truncated_normal([14*14*32, 10], 0.0, 0.01), name="out_weight")

        self.out = tf.nn.softmax(tf.matmul(fc, out_weight))

        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.out))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def train(self, sess, batch_x, batch_y):

        _, accuracy = sess.run([self.train_step, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y})

        return accuracy

        # return self.calc_accuracy(sess, batch_x, batch_y)


    def calc_accuracy(self,sess, batch_x, batch_y):
        correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return sess.run(accuracy, feed_dict={self.x: batch_x, self.y: batch_y})





if __name__ == '__main__':

     model = MnistCNN()

     with tf.Session() as sess:

         sess.run(tf.global_variables_initializer())

         for i in range(1000):
             batch = mnist.train.next_batch(50)

             accuracy = model.train(sess, batch[0], batch[1])
             if i % 100 == 0:
                print("step %d, training accuracy %g" % (i, accuracy))




