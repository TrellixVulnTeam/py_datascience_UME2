import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class MNIST_Training_simple:
    def __init__(self):    
        mnist = input_data.read_data_sets( " MNIST_data/ " , one_hot=True)
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        print(W)
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        print(y)
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initializers.global_variables()

        sess = tf.Session()
        sess.run(init)

        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x:batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Точность %s", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

class MNIST_Training_ReLU:
    def __init__(self):    
        mnist = input_data.read_data_sets( " MNIST_data/ " , one_hot=True)
        #1st Layer
        x = tf.placeholder(tf.float32, [None, 784])

        #Hidden Layer
        W_relu = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
        b_relu = tf.Variable(tf.truncated_normal([100], stddev=0.1))
        h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)
        keep_probability = tf.placeholder(tf.float32)
        h_drop = tf.nn.dropout(h, keep_probability)

        #3rd layer
        W = tf.Variable(tf.zeros([100, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(h_drop, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 10])        
       # keep_probability = tf.placeholder(tf.float32)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initializers.global_variables()

        sess = tf.Session()
        sess.run(init)

        for i in range(2000):
            print(i)
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print( " Точность: %s " % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_probability: 1.} ))

