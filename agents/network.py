import tensorflow as tf
import numpy as np
from copy import deepcopy
from collections import deque, namedtuple

class Network():
    def __init__(self, sess, name="default"):
        self.sess = sess
        self.copy_op = None
        self.name = name
        self.vars = {}

    def build_model(self, input_shape, output_shape, name):

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(dtype=tf.float32, shape = [None,]+input_shape, name="input")

            x = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu)
            x = tf.layers.dense(x, 16, activation=tf.nn.relu)
            x = tf.layers.dense(x, 16, activation=tf.nn.relu)
            self.outputs = tf.layers.dense(x , output_shape)

            for v in tf.trainable_variables(scope=name):
                self.vars[v.name] = v

        return self.inputs, self.outputs, 
