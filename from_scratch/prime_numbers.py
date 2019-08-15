from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

#tf.enable_eager_execution()

class PrimeNumbersClassifier:
    def __init__(self): 
        # self.writer = tf.summary.FileWriter('.')
        # self.writer.add_graph(tf.get_default_graph())
        # self.writer.flush()
        datas = self.createData()
        #print(datas.shuffle(1000).batch(1000))
        # for feat, targ in datas.take(5):
        #   print ('Features: {}, Target: {}'.format(feat, targ))
        self.startTrain(datas)

    def createData(self):
        # csv_file = tf.keras.utils.get_file('heart.csv', '/media/user/Transcend/AI/Datasets/primes1.txt')
        # raw_train_data = get_dataset(train_file_path)
        # raw_test_data = get_dataset(test_file_path)
        prime_numbers = pd.read_csv('/media/user/Transcend/AI/Datasets/primes1.txt', header=None, delim_whitespace=True)
        prime_numbers = prime_numbers.values.flatten()
        prime_numbers = pd.Series(prime_numbers.transpose())
        prime_numbers_and_not = pd.DataFrame({
            'PM': prime_numbers,
            'is_prime': pd.Series(1)
        })
        datas_to_save = pd.DataFrame({
            'Number': range(0, prime_numbers_and_not['PM'].iloc[-1]+1)
        })
        datas_to_save['is_prime'] = datas_to_save['Number'].isin(prime_numbers_and_not['PM'])
        print(datas_to_save)
        training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(datas_to_save['Number'].values, tf.int64), tf.cast(datas_to_save['is_prime'].values, tf.int64)))
        return training_dataset

    def saveData(self, datas):
        datas.to_csv('/media/user/Transcend/AI/Datasets/primes.csv')
    def startTrain(self, dataset):
        # we will load 1 number
        x = tf.placeholder(tf.float32, [None, 1])
        # each number can have 1 of 2 classes
        W = tf.Variable(tf.zeros([1, 2]))
        b = tf.Variable(tf.zeros([2]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 1])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initializers.global_variables()
        sess = tf.compat.v1.Session()
        sess.run(init)
        for i in range(3):
            datas = dataset.shuffle(3).batch(1000)
            iterator = datas.make_one_shot_iterator()
            batch_xs, batch_ys = iterator.get_next()
            batch_xs = sess.run(batch_xs)
            batch_xs = batch_xs.reshape(1000, 1)
            batch_ys = sess.run(batch_ys)
            batch_ys = batch_ys.reshape(1000, 1)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Точность %s", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    