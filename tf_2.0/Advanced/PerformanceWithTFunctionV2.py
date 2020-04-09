# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/8'


import traceback
import contextlib
import tensorflow as tf

@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}:'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))








