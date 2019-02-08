import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils

class NormalizedInitialization:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)        
        X_train = x_train.reshape([-1, 28*28]) / 255.
        X_test = x_test.reshape([-1, 28*28]) / 255.

        uniform_model = self.create_model("uniform" )
        uniform_model.compile(loss= 'categorical_crossentropy' , optimizer = 'sgd' , metrics = ['accuracy'])
        uniform_model.fit(x_train, Y_train, batch_size=64, nb_epoch=30, verbose=1, validation_data=(x_test, Y_test))

        glorot_model = self.create_model( "glorot_normal" )
        glorot_model.compile(loss= 'categorical_crossentropy' , optimizer= 'sgd' , metrics=[ 'accuracy' ])
        glorot_model.fit(x_train, Y_train,
            batch_size=64, nb_epoch=30, verbose=1, validation_data=(x_test, Y_test))

    def create_model(self, init):
        model = Sequential()
        model.add(Dense(100, input_shape=(28*28,), init=init, activation= 'tanh' ))
        model.add(Dense(100, init=init, activation= 'tanh' ))
        model.add(Dense(100, init=init, activation= 'tanh' ))
        model.add(Dense(100, init=init, activation= 'tanh' ))
        model.add(Dense(10, init=init, activation= 'softmax' ))
        return model

    