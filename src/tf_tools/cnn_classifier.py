
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras


# conv_dropout = 0.25
# pool_window = 8
# ksize = 10
# n_fc = 32
# n_filters = 128

def getClassCNN(seq_len, n_classes, n_filters=128, n_fc=32, ksize=10, poolW=8, dropout=0.25, lr=0.0001):
    
    model = keras.models.Sequential([

        keras.layers.Conv1D(filters=n_filters, kernel_size=ksize, strides=1, padding="same", input_shape=[seq_len,4], name="conv_1"),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Activation(keras.activations.relu, name="relu_1"),
        keras.layers.Dropout(dropout,name="dropout_1"),

        keras.layers.Conv1D(filters=128, kernel_size=ksize, strides=1, padding="same", name="conv_2"),
        keras.layers.BatchNormalization(name="batch_norm_2"),
        keras.layers.Activation(keras.activations.relu, name="relu_2"),
        keras.layers.Dropout(dropout, name="dropout_2"),

        keras.layers.MaxPooling1D(pool_size = poolW, name="max_pool_1"),

        keras.layers.Conv1D(filters=n_filters, kernel_size=ksize, strides=1, padding="same", name="conv_3"),
        keras.layers.BatchNormalization(name="batch_norm_3"),
        keras.layers.Activation(keras.activations.relu, name="relu_3"),


        keras.layers.MaxPooling1D(pool_size= poolW, name="max_pool_2"),
        keras.layers.Dropout(dropout, name="dropout_3"),
         
        ########################################################################

        keras.layers.Flatten(name="flatten_1"),
        keras.layers.Dense(n_fc, name="dense_1"),
        keras.layers.BatchNormalization(name="batch_norm_4"),
        keras.layers.Activation(keras.activations.relu, name="relu_4"),
        keras.layers.Dropout(dropout, name="dropout_4"),
        keras.layers.Dense(n_classes, activation="softmax", name="dense_2"),
    ])
    
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr),
              metrics=['accuracy'])

    return model
