import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation

class SimpleNormalDistribution:
    def __init__(self):
        logr = Sequential()
        logr.add(Dense(1, input_dim=2, activation='sigmoid'))
        logr.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        X_train, Y_train = self.sample_data()
        X_test, Y_test = self.sample_data(100)
        logr.fit(X_train, Y_train, batch_size=16, epochs=100, verbose=1, validation_data=(X_test, Y_test))
        
    def sampler(self, n, x, y):
        return np.random.normal(size=[n,2]) + [x, y]

    def sample_data(self, n=1000, p0=(-1., -1.), p1=(1., 1.)):
        zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
        labels = np.vstack([zeros, ones])
        z_sample = self.sampler(n, x=p0[0], y=p0[1])
        o_sample = self.sampler(n, x=p1[0], y=p1[1])
        return np.vstack([z_sample, o_sample]), labels

    