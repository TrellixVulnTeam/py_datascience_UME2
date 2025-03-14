from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

#tf.enable_eager_execution()

class TFPrimeNumbersClassifier:
    def __init__(self): 
        # self.writer = tf.summary.FileWriter('.')
        # self.writer.add_graph(tf.get_default_graph())
        # self.writer.flush()
        datas = self.createData()
        #print(datas.shuffle(1000).batch(1000))
        # for feat, targ in datas.take(5):
        #   print ('Features: {}, Target: {}'.format(feat, targ))
        self.startTrain(datas)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def createData(self):
        # csv_file = tf.keras.utils.get_file('heart.csv', '/media/user/Transcend/AI/Datasets/primes1.txt')
        # raw_train_data = get_dataset(train_file_path)
        # raw_test_data = get_dataset(test_file_path)
        prime_numbers = pd.read_csv('C:\\Users\\user\\py_datascience\\Datasets\\primes1.txt', header=None, delim_whitespace=True)
        prime_numbers = prime_numbers.values.flatten()
        prime_numbers = pd.Series(prime_numbers.transpose())
        prime_numbers_and_not = pd.DataFrame({
            'PM': prime_numbers
        })
        datas_to_save = pd.DataFrame({
            'Number': range(0, prime_numbers_and_not['PM'].iloc[-1]+1),
            'is_prime': [ (np.array([0,1]))  for i in range(0, prime_numbers_and_not['PM'].iloc[-1]+1) ],
            'no_prime': np.zeros(prime_numbers_and_not['PM'].iloc[-1]+1),
            'yes_prime': np.zeros(prime_numbers_and_not['PM'].iloc[-1]+1)
        })
        datas_to_save.loc[datas_to_save['Number'].isin(prime_numbers_and_not['PM']),'yes_prime'] = 1
        datas_to_save.loc[~datas_to_save['Number'].isin(prime_numbers_and_not['PM']),'no_prime'] = 1
        
#        datas_to_save.loc[not datas_to_save['Number'].isin(prime_numbers_and_not['PM']), 'is_prime'] = pd.Series([0, 1])
        training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(datas_to_save['Number'].values, tf.float32), tf.cast(datas_to_save['yes_prime'].values, tf.float32), tf.cast(datas_to_save['no_prime'].values, tf.float32)))
        return training_dataset

    def saveData(self, datas):
        datas.to_csv('/media/user/Transcend/AI/Datasets/primes.csv')
    def startTrain(self, dataset):
        # we will load 1 number
        x = tf.placeholder(tf.float32, [None, 1000])
        # each number can have 1 of 2 classes
        W = tf.Variable(tf.zeros([1000, 2]))
        b = tf.Variable(tf.zeros([2]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 2])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initializers.global_variables()
        sess = tf.compat.v1.Session()
        sess.run(init)
        datas = dataset.shuffle(1000).batch(1000)
        iterator = datas.make_one_shot_iterator()
        for i in range(15):
            print("batch: ", i)
            batch_xs, batch_ys1, batch_ys2 = iterator.get_next()
            batch_xs = sess.run(tf.reshape(batch_xs, [1, 1000]))
            batch_ys1 = sess.run(tf.reshape(batch_ys1, [1000, 1]))
            #print(batch_ys1)
            batch_ys2 = sess.run(tf.reshape(batch_ys2, [1000, 1]))
            #print(bath_ys2)
            batch_ys = sess.run(tf.concat([batch_ys1, batch_ys2], 1))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        batch_xs, batch_ys1, batch_ys2 = iterator.get_next()
        batch_xs = sess.run(tf.reshape(batch_xs, [1, 1000]))
        batch_ys1 = sess.run(tf.reshape(batch_ys1, [1000, 1]))
        #print(batch_ys1)
        batch_ys2 = sess.run(tf.reshape(batch_ys2, [1000, 1]))
        #print(bath_ys2)
        batch_ys = sess.run(tf.concat([batch_ys1, batch_ys2], 1))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(correct_prediction.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        print(accuracy)
        # print("Точность %s", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    
class KerasPrimeNumbersClassifier:
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
        prime_numbers = pd.read_csv('.\\Datasets\\primes1.txt', header=None, delim_whitespace=True)
        prime_numbers = prime_numbers.values.flatten()
        prime_numbers = pd.Series(prime_numbers.transpose())
        prime_numbers_and_not = pd.DataFrame({
            'PM': prime_numbers
        })
        datas_to_save = pd.DataFrame({
            'Number': range(0, prime_numbers_and_not['PM'].iloc[-1]+1),
            'is_prime': [ (np.array([0,1]))  for i in range(0, prime_numbers_and_not['PM'].iloc[-1]+1) ],
            'no_prime': np.zeros(prime_numbers_and_not['PM'].iloc[-1]+1),
            'yes_prime': np.zeros(prime_numbers_and_not['PM'].iloc[-1]+1)
        })
        datas_to_save.loc[datas_to_save['Number'].isin(prime_numbers_and_not['PM']),'yes_prime'] = 1
        datas_to_save.loc[~datas_to_save['Number'].isin(prime_numbers_and_not['PM']),'no_prime'] = 1
        
#        datas_to_save.loc[not datas_to_save['Number'].isin(prime_numbers_and_not['PM']), 'is_prime'] = pd.Series([0, 1])
        training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(datas_to_save['Number'].values, tf.float32), tf.cast(datas_to_save['yes_prime'].values, tf.float32), tf.cast(datas_to_save['no_prime'].values, tf.float32)))
        return training_dataset

    def saveData(self, datas):
        datas.to_csv('.\\Datasets\\primes.csv')
    def startTrain(self, dataset):
        # we will load 1 number
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1000, activation='softmax', input_shape=(1000,)))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        datas = dataset.shuffle(1000).batch(1000)
        iterator = datas.make_one_shot_iterator()
        batch_xs, batch_ys1, batch_ys2 = iterator.get_next()
        sess = tf.Session()
        batch_xs = sess.run(tf.reshape(batch_xs, [1, 1000]))
        batch_ys1 = sess.run(tf.reshape(batch_ys1, [1, 1000]))
        #print(batch_ys1)
        batch_ys2 = sess.run(tf.reshape(batch_ys2, [1, 1000]))
        #print(bath_ys2)
        batch_ys = sess.run(tf.concat([batch_ys1, batch_ys2], 1))
        model.fit(x=batch_xs, y=batch_ys,
                steps_per_epoch=100,
                epochs=20)
        score = model.evaluate(batch_xs, batch_ys, batch_size=1000)

        print(accuracy)
        # print("Точность %s", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    