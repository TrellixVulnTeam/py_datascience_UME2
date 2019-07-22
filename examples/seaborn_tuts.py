from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from time import time
import seaborn as sns

def show_tutorial():
    tips = sns.load_dataset("tips")
    sns.violinplot(x = "total_bill", data = tips)
    plt.show()

def show_iris():
    iris = sns.load_dataset("iris")
    sns.swarmplot(x = "species", y = "petal_length", data = iris)
    plt.show()
