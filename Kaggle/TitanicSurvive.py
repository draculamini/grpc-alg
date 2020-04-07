# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/7'


import pandas as pd
import numpy as np

with open("data/titanic/train.csv") as f:
    lines = f.readlines()
    print(lines)

train_path = "data/titanic/train.csv"
test_path = "data/titanic/test.csv"