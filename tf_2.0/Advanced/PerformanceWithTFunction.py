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



# @tf.function
# def add(a, b):
#     return a + b
#
# print(add(tf.ones([2, 2]), tf.ones([2, 2])))
#
# @tf.function
# def dense_layer(x, w, b):
#     return add(tf.matmul(x, w), b)
#
# print(dense_layer(tf.ones([2, 3]), tf.ones([3, 2]), tf.ones([2, 2])))
#
# @tf.function
# def double(a):
#     return a + a

# print(double(tf.constant(1)))
# print()
# print(double(tf.constant(1.1)))
# print()
# print(double(tf.constant("a")))
# print()
#
#
# print("Obtaining concrete trace")
#
# double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
# print("Executing traced function")
# print(double_strings)
# print(double_strings(tf.constant("a")))
# print(double_strings(a=tf.constant("b")))
# print("Using a concrete trace with incompatible types will throw an error")
#
# with assert_raises(tf.errors.InvalidArgumentError):
#   double_strings(tf.constant(1))
#

# def train_one_step():
#     pass
#
#
# @tf.function
# def train(num_steps):
#     print("Tracing with num_steps = {}".format(num_steps))
#     for _ in tf.range(num_steps):
#         train_one_step()
#
# train(num_steps=10)
# train(num_steps=20)
#
# train(num_steps=tf.constant(10))
# train(num_steps=tf.constant(20))
#
#
# @tf.function
# def f(x):
#     print("Traced with", x)
#     tf.print("Executed with", x)
# f(1)
# f(1)
# f(2)


# @tf.function
# def f(x):
#     v = tf.Variable(1.0)
#     v.assign_add(x)
#     return v
#
# with assert_raises(ValueError):
#     f(1.0)

# @tf.function
# def f(x):
#     v = tf.Variable(1.0)
#
#     return v.assign_add(x)
#
# print(f(1.0))
# print(f(5.0))

# class C:
#     pass
#
# obj = C()
# obj.v = None
#
# @tf.function
# def g(x):
#     if obj.v is None:
#         obj.v = tf.Variable(1.0)
#
#     return obj.v.assign_add(x)
#
# print(g(1.0))
# print(g(3.0))

# state = []
# @tf.function
# def fn(x):
#     if not state:
#         state.append(tf.Variable(2.0 * x))
#         state.append(tf.Variable(state[0] * 3.0))
#     return state[0] * x * state[1]
#
# print(fn(tf.constant(1.0)))
# print(fn(tf.constant(3.0)))

# @tf.function
# def f(x):
#     while tf.reduce_sum(x) > 1:
#         tf.print(x)
#         x = tf.tanh(x)
#     return x
# f(tf.random.uniform([5]))


# def f(x):
#     while tf.reduce_sum(x) > 1:
#         tf.print(x)
#         x = tf.tanh(x)
#     return x
# print(tf.autograph.to_code(f))


# def test_tf_cond(f, *args):
#     g = f.get_concrete_function(*args).graph
#     if any(node.name == 'cond' for node in g.as_graph_def().node):
#         print("{}({}) uses tf.cond.".format(
#             f.__name__, ', '.join(map(str, args))))
#     else:
#         print("{}({}) executes normally.".format(
#             f.__name__, ', '.join(map(str, args))))
#
#     print("  result: ", f(*args).numpy())
#
#
# @tf.function
# def dropout(x, training=True):
#
#     if training:
#         x = tf.nn.dropout(x, rate=0.5)
#     return x
#
# # test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), True)
#
#
# @tf.function
# def f(x):
#     if x > 0:
#         x = x + 1
#         print("Tracing 'then' branch")
#     else:
#         x = x - 1
#         print("Tracing 'else' branch")
#     return x
#
# print(f(-1.0).numpy())
#
# print(f(tf.constant(1.0)).numpy())


# @tf.function
# def f():
#     if tf.constant(True):
#         x = tf.ones([3, 3])
#     return x
#
# with assert_raises(ValueError):
#     f()

# @tf.function
# def f(x, y):
#     if bool(x):
#         y = y + 1
#         print("Tracing 'then' branch")
#     else:
#         y = y - 1
#         print("Tracing `else` branch")
#
#     return y
#
# print(f(True, 0).numpy())

def test_dynamically_unrolled(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'while' for node in g.as_graph_def().node):
    print("{}({}) uses tf.while_loop.".format(
        f.__name__, ', '.join(map(str, args))))
  elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
    print("{}({}) uses tf.data.Dataset.reduce.".format(
        f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) gets unrolled.".format(
        f.__name__, ', '.join(map(str, args))))

# @tf.function
# def for_in_range():
#   x = 0
#   for i in range(5):
#     x += i
#   return x
#
# test_dynamically_unrolled(for_in_range)
#
# @tf.function
# def for_in_tfrange():
#   x = tf.constant(0, dtype=tf.int32)
#   for i in tf.range(5):
#     x += i
#   return x
# test_dynamically_unrolled(for_in_tfrange)

batch_size = 2
seq_len = 3
feature_size = 4


def rnn_step(inp, state):
    return inp + state


@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
    input_data = tf.transpose(input_data, [1, 0, 2])
    print(input_data)
    states = tf.TensorArray(tf.float32, size=max_seq_len)
    print("states", states)
    state = initial_state
    for i in tf.range(max_seq_len):
        state = rnn_step(input_data[i], state)
        states = states.write(i, state)
    return tf.transpose(states.stack(), [1, 0, 2])


print(dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size])))












