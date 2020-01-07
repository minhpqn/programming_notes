"""Check if GPU can be used in your machine
Reference:
[https://www.quora.com/How-do-I-confirm-that-my-TensorFlow-scripts-are-running-in-GPU-when-using-Python-script](https://www.quora.com/How-do-I-confirm-that-my-TensorFlow-scripts-are-running-in-GPU-when-using-Python-script)
"""
import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))