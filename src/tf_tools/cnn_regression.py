import pandas as pd
import numpy as np
import random
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model

dropout = 0.15

class MCDropout(layers.Dropout):
    def call(self,inputs):
        return super().call(inputs, training=True)


def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = layers.Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2c')(X)

    X = layers.Add()([X, X_shortcut])# SKIP Connection
    X = layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2c')(X)

    X_shortcut = layers.Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1')(X_shortcut)
    X_shortcut = layers.BatchNormalization(name=bn_name_base + '1')(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

## As described in Ryan Friedman enhancer_resnet
"""
    A DilationBlock is a Sequence of convolutions with exponentially increasing dilations. Each convolution is
    followed by batch normalization, activation, and dropout. The final convolution is followed by a batch
    normalization step but not activation or dropout. The entire block is surrounded by a skip connection to create a
    ResBlock. The DilationBlock should be followed by an activation function and dropout after the skip connection.
"""
def DilationBlock(X, nfilters, nlayers, filter_size, basename, rate = 2, activation='relu', dropout=0.1):
    X_shortcut = X
    # Initial Layer
    X = layers.Conv1D(nfilters, filter_size, padding='same', name='_'.join(['di',basename,'conv','0']))(X)
    X = layers.BatchNormalization( name='_'.join(['di',basename,'bn','0']))(X)
    
    for dilation in range(1,nlayers):
        X = layers.Activation(activation)(X)
        X = MCDropout(dropout, name='_'.join(['di',basename,'mcdrop',str(dilation)]))(X)
        X = layers.Conv1D(filters=nfilters,kernel_size=filter_size, padding='same', dilation_rate=rate**dilation)(X)
        X = layers.BatchNormalization(name='_'.join(['di',basename,'bn',str(dilation)]))(X)
        
    X = layers.Add()([X, X_shortcut]) # SKIP Connection
    return X

def ResNet50(input_shape=(164,4)):

    X_input = Input(input_shape)

    X = layers.ZeroPadding1D(3)(X_input)

    X = layers.Conv1D(64, 7, strides=2, name='conv1')(X)
    X = layers.BatchNormalization( name='bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = MCDropout(dropout, name='dropout1')(X)
    X = layers.MaxPooling1D(3, strides=2)(X)
    

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = MCDropout(dropout, name='dropout2')(X)

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = MCDropout(dropout, name='dropout3')(X)

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = MCDropout(dropout, name='dropout4')(X)

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = MCDropout(dropout, name='dropout5')(X)

    X = layers.AveragePooling1D(2, padding='same')(X)
    
    X = layers.Flatten()(X)
    X = layers.Dense(256, activation='relu')(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(1, activation='linear')(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def smallerResNet(input_shape=(164,4),
                  init_block={'num_filters': 128, 'filter_size': 10, 'pool_size': 4, 'strides': 1}):

    X_input = Input(input_shape)

    X = layers.ZeroPadding1D(init_block['filter_size'] // 2)(X_input)

    X = layers.Conv1D(filters=init_block['num_filters'],
                      kernel_size=init_block['filter_size'],
                      strides=init_block['strides'], name='conv1')(X)
    X = layers.BatchNormalization( name='bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = MCDropout(dropout, name='dropout1')(X)
    X = layers.MaxPooling1D(3, strides=2)(X)
    

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = MCDropout(dropout, name='dropout2')(X)

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = MCDropout(dropout, name='dropout3')(X)

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = MCDropout(dropout, name='dropout4')(X)

    X = layers.AveragePooling1D(4, padding='same')(X)
    
    X = layers.Flatten()(X)
    X = layers.Dense(256, activation='relu')(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(1, activation='linear')(X)
    
    model = Model(inputs=X_input, outputs=X, name='smallerResNet')

    return model

def originalResNet(input_shape=(164,4), lr = 0.0002,output='linear'):
    
    X_input = Input(input_shape)
    X = X_input
    
    architecture = [
        ['A', 128, 10, "exponential", 4, 4],   # 164 --> 41
        ['B', 128, 6, "relu", 3, 4],    # 41 --> 11
        ['C', 64, 3, "relu", 2, 3],   # 11 --> 4
    ]
    conv_dropout = 0.1
    resid_dropout = 0.25
    dilation_filter_size = 3
    fc_neurons = 64
    fc_dropout = 0.25
    
    for name, nfilters, filter_size, activation, dilation_layers, pool_size in architecture:
        X = layers.ZeroPadding1D(filter_size-1)(X)
        X = layers.Conv1D(filters=nfilters, kernel_size=filter_size,padding='valid')(X)
        X = layers.BatchNormalization(name=name+'_bn')(X)
        X = layers.Activation(activation)(X)
        X = MCDropout(conv_dropout, name=name+'_cMCdrop')(X)
        
        X = DilationBlock(X,nfilters,dilation_layers,dilation_filter_size,name)
        X = layers.Activation('relu')(X)
        X = MCDropout(resid_dropout, name=name+'_rMCdrop')(X)
        X = layers.MaxPool1D(pool_size=pool_size,strides=pool_size)(X)
        
    X = layers.Flatten()(X)
    X = layers.Dense(fc_neurons, name="Dense_1")(X)
    X = layers.BatchNormalization(name="Linear_bn")(X)
    X = layers.Activation('relu')(X)
    X = MCDropout(fc_dropout, name='fc_MCdrop')(X)
    X = layers.Dense(1, activation=output, name="output_layer")(X)
    
    model = Model(inputs=X_input, outputs=X, name='originalResNet')
    
    model.compile(loss='mae',optimizer=keras.optimizers.Adam(learning_rate=lr))

    return model

def tranferNet(input_shape=(164,4), lr = 0.0002,output='linear'):
    
    hidden_layers = 5
    dense_layers = 2
    neurons = 180
    kernels = [11,7,3,3,3]
    dense_neurons = 128
    pool_size=3
    dropout=0.4
    
    model = keras.Sequential()
    model.add(Input(input_shape))
    
    for i in range(hidden_layers):
        model.add(layers.Conv1D(filters=neurons,kernel_size=kernels[i],padding='same',name='conv_'+str(i)))
        model.add(layers.BatchNormalization(name='conv_bn_'+str(i)))
        model.add(layers.Activation('relu'))
        
    model.add(layers.MaxPooling1D(pool_size=pool_size, name='pool_'+str(i)))
    model.add(layers.Flatten())
    for i in range(dense_layers):
        model.add(layers.Dense(units=dense_neurons))
        model.add(layers.BatchNormalization(name='dense_bn_'+str(i)))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(dropout, name='MCD_'+str(i)))
    model.add(layers.Dense(units=1, activation=output, name='output'))
    
    model.compile(loss='mae',optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model

def bestResNet(input_shape=(164,4), lr = 0.0002, conv_k =[(11,128),(5,128),(3,64)], dense_n=[256,128], dropout=0.1,
               pool_size=3, bins=1,output='linear',loss='mae',optimizer=keras.optimizers.Adam):
    
    #Hardcoding 2 conv per block, and 2 dense layers, may change
    block = 1
    
    model = keras.Sequential()
    model.add(Input(input_shape))
    
    for size, shape in conv_k:
        model.add(layers.Conv1D(filters=shape,kernel_size=size,padding='same',name=str(block)+'_conv1'))
        model.add(layers.Conv1D(filters=shape,kernel_size=size,padding='same',name=str(block)+'_conv2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size=pool_size))
        block += 1
    
    model.add(layers.Flatten())
    for neurons in dense_n:
        model.add(layers.Dense(units=neurons))
        model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(bins,activation=output))
    
    model.compile(loss=loss,optimizer=optimizer(learning_rate=lr))
    return model
    
        
