# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/29'


import tensorflow as tf


class YoutubeNN(tf.keras.Model):

    def __init__(self, train_data, mode=None, embedding_dim=64,
                 window_size = 5, min_count=10, neg_sample_cnt=10,
                 dense_layers=[]):
        super(YoutubeNN, self).__init__()

        if mode == 'train':
            pass
        elif mode == 'test':
            pass



    def call(self, inputs):
        pass

if __name__ == '__main__':
    pass