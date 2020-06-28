# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/17'


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

path = "/Users/wuxikun/Downloads/longMovie.txt"

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
