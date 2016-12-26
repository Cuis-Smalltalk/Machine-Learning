#!/usr/bin/python3
import tensorflow as tf
import numpy as np

BASE = 'GraphDefs'

def writeg(name):
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    tf.train.write_graph(session.graph_def, BASE, name + '.txt', True)
    tf.train.write_graph(session.graph_def, BASE, name, False)

    tf.reset_default_graph()
    del session
    print (name)

writeg('empty.pb')

a = tf.constant(42, name='a')
writeg('a=42.pb')

a = tf.constant(0x4242424242424242, name='a')
writeg('a=4242424242424242.pb')

a = tf.constant(0.42, name='a')
writeg('a=0.42.pb')

a = tf.Variable(6.0, name='a')
b = tf.Variable(7.0, name='b')
c = tf.mul(a, b, name="c")
writeg('c=(a=6.0)+(b=7.0).pb')

a = tf.Variable(6, name='a')
b = tf.Variable(7, name='b')
c = tf.mul(a, b, name="c")
writeg('c=(a=6)+(b=7).pb')


