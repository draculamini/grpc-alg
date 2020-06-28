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

@tf.function
def add(a, b):
    return a + b

print(add(tf.ones([2, 2]), 3))
print(add(tf.ones([2, 2]), tf.ones([2, 2])))

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
print(tape.gradient(result, v))


@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

print(dense_layer(tf.ones([2, 3]), tf.ones([3, 3]), 4))


@tf.function
def double(a):
    print("Tracing with", a)
    return a + a
print(double(1))
print(double(tf.constant(1.0)))
print(double(tf.constant('c')))

print("------------")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("a")))

print("Executing traced function")
# print(double_strings(tf.constant("a")))
# print(double_strings(tf.constant("b")))
# print("Using a concrete trace with incompatible types will throw")
# with assert_raises(tf.errors.InvalidArgumentError):
#     double_strings(tf.constant(1))


# @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
# def next_collatz(x):
#     print("Tracing with", x)
#     return tf.where(x%2 == 0, x)
#
# print(next_collatz(tf.constant([1, 2])))
# # We specified a 1-D tensor in the input signature, so this should fail.
# with assert_raises(ValueError):
#   next_collatz(tf.constant([[1, 2], [3, 4]]))

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x//2, 3*x + 1)
print(next_collatz(tf.constant([1, 2])))
# with assert_raises(ValueError):
#     next_collatz(tf.constant([[1, 2], [3, 4]]))


def train_one_step():
    pass


@tf.function
def train(num_steps):
    print("Tracing with num_steps = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()
train(num_steps=10)
train(num_steps=20)

train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))


@tf.function
def f(x):
    print("Traced with", x)
    tf.print("Executed with ", x)

f(1)
f(1)
f(2)

external_list = []

print("---------")


def side_effect(x):
    print("Python side effect")
    external_list.append(x)


@tf.function
def f(x):
    tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)

assert len(external_list) == 3
assert external_list[0].numpy() == 1

external_var = tf.Variable(0)


@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var", external_var)


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(
        f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x)
    return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10

print(small_data)
print(big_data)

measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))


a = tf.Variable(1.0)
b = tf.Variable(2.0)
a.assign(2.0 * 2.0)
print(a)

b.assign_add(a)
print(b)

v = tf.Variable(1.0)


@tf.function
def f(tmp, x):
    return tmp.assign_add(x)


print(f(v, 1.0))
print(f(v, 2.0))

# with assert_raises(ValueError):
#     f(1.0)

class C:
    pass

obj = C()
obj.v = None


@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)

print(g(1.0))
print(g(2.0))


state = []
@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))

    return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))


# @tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

f(tf.random.uniform([5]))

print(tf.autograph.to_code(f))


def test_tf_cond(f, *args):

    g = f.get_concrete_function(*args).graph
    if any(node.name == 'cond' for node in g.as_graph_def().node):
        print("{}({}) uses tf.cond.".format(
        f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) executes normally.".format(
        f.__name__, ', '.join(map(str, args))))

    print("  result: ",f(*args).numpy())


@tf.function
def dropout(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), True)

@tf.function
def f(x):
    if x > 0:
        print("Tracing `then` branch")
        x = x + 1
    else:
        print("Tracing `else` branch")
        x = x - 1
    return x

print(f(-1.0).numpy())
print(f(1.0).numpy())
print(f(tf.constant(1.0)).numpy())


# @tf.function
# def f():
#     if tf.constant(True):
#         x = tf.ones([3, 3])
#     return x

# f()

# with assert_raises(ValueError):
#     f()


@tf.function
def f(x, y):
    if bool(x):
        y = y + 1.
        print("Tracing 'then' branch")
    else:
        y = y - 1.
        print("Tracing 'else' branch")
    return y

print(f(True, 0).numpy())


# f(tf.constant(True), 0.0)
# with assert_raises(TypeError):
#     f(True, 0.0)


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


@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_range)


@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_tfrange)

# print(tf.autograph.to_code(for_in_tfrange))


@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x>0:
        x -= 1
    return x

test_dynamically_unrolled(while_tf_cond)


@tf.function
def buggy_while_py_true_tf_break(x):
    while True:
        if tf.equal(x, 0):
            break
        x -= 1
    return x

# with assert_raises(TypeError):
#     test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)


@tf.function
def while_tf_true_tf_break(x):
    while tf.constant(True):
        if x == 0:
            break
        x -= 1
    return x

# with assert_raises(TypeError):
#     test_dynamically_unrolled(while_tf_true_tf_break, 5)


@tf.function
def tf_for_py_break():
    x = 0
    for i in tf.range(5):
        if i == 3:
            break
        x += i
    return x
test_dynamically_unrolled(tf_for_py_break)


batch_size = 2
seq_len = 3
feature_size = 4


def rnn_step(inp, state):
    return inp + state






