# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/30'


import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, name="x")

w1 = tf.Variable(20.0, name="w1")
w2 = tf.multiply(w1, x, name="w2")
b1 = tf.Variable(2.0,name="bias")
out = tf.add(w2, b1, name="out")

modelPath = 'model/wx_b2/'
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["out"])
#     tf.train.write_graph(graph, '.', modelPath, as_text=False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


builder = tf.saved_model.builder.SavedModelBuilder(modelPath)

x_tensor = tf.saved_model.utils.build_tensor_info(x)
out_tensor = tf.saved_model.utils.build_tensor_info(out)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': x_tensor},
        outputs={'out': out_tensor},  #,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    ))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()
