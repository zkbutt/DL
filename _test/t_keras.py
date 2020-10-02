import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from f_tools.f_gen import SimilarityGenerator
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed, Add
from tensorflow.keras.layers import Activation, Flatten

if __name__ == '__main__':
    inputs = Input(shape=(600, 600, 3))
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    model = Model(inputs, x)
    print(model.summary())
