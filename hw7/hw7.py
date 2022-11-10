import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, Dense, LeakyReLU, Input, GRU, Embedding, Conv1D, MaxPooling1D,MultiHeadAttention,Input, GlobalAveragePooling1D
from tensorflow.keras import backend
from tensorflow.keras.utils import plot_model
from statistics import mean
import pickle

import random

import re
import pfam_loader
import argparse

##################

def attention_CNN_network(n_tokens,len_max,ins_train,outs_train,spa_accuracy,
                   n_neurons, 
                   activation, 
                   activation_dense,
                   cnn_filter,
                   cnn_kernel,
                   cnn_pool,
                   dense_unit, 
                   lambda_regularization):
    
    # intializing the lambda
    if lambda_regularization is not None:
        lambda_regularization=tf.keras.regularizers.l2(lambda_regularization)
    
    #model = Sequential()
    #input layer
    Input_layer = Input(shape =(len_max,))
    
    #Embedding layer
    new = Embedding(n_tokens,15, input_length=len_max)(Input_layer)
    
    #CNN layer
    for i in range(len(cnn_filter)):
        new = Conv1D(filters=cnn_filter[i], kernel_size=cnn_kernel[i],activation=activation_dense, padding ='same')(new)
        if ((i+1)%2 == 0):
            new = MaxPooling1D(pool_size=cnn_pool[i])(new)
    
    
    # MultiHeadAttention layer
    new = MultiHeadAttention(num_heads=2,
                             key_dim=new.shape[1],
                             use_bias=True,
                             kernel_initializer='random_uniform',
                             bias_initializer='random_uniform',
                             kernel_regularizer=lambda_regularization)(new,new)

    new = GlobalAveragePooling1D()(new)
    
    #dense layer
    for i in range(len(dense_unit)):
        new = Dense(units=dense_unit[i],
                      activation=activation_dense,
                      use_bias=True,
                      kernel_initializer='random_uniform',
                      bias_initializer='random_uniform',
                      kernel_regularizer=lambda_regularization)(new)
    
    
    #output layer
    new = Dense(units=46,
                activation='softmax',
                use_bias=True,
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                kernel_regularizer=lambda_regularization)(new)
    
    model = Model(Input_layer, new, name='Multi_head_attention')
    
    
    # The optimizer determines how the gradient descent is to be done
    opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
    
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=[spa_accuracy])
    
    return model

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Multi_head_attention', fromfile_prefix_chars='@')
    
    # High-level experiment configuration
    parser.add_argument('--name', type=str, default=None, help="name of the model")
    parser.add_argument('--rotation', type=int, default=0, help='rotation')
    
    #parameters for building model
    parser.add_argument('--conv_kernel', nargs='+', type=int, default=[3,5], help='Convolution kernel size size per layer')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--dense', nargs='+', type=int, default=[100, 5], help='Number of dense units per layer')
        
    return parser

if __name__ == "__main__":
        
    parser = create_parser()
    args = parser.parse_args()
    
    
    fi = []
    eva = []
    

    # splits the dataset into train, validation and test
    dat_out = pfam_loader.prepare_data_set(rotation = args.rotation)

        
    #initializing the metrics
    spa_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
    #building the model
    model = attention_CNN_network(dat_out['n_tokens'],
                            dat_out['len_max'],
                            dat_out['ins_train'],
                            dat_out['outs_train'],
                            spa_accuracy, 
                            n_neurons=40, 
                            activation='tanh', 
                            activation_dense='elu',
                            cnn_filter=args.conv_nfilters,
                            cnn_kernel=args.conv_kernel,
                            cnn_pool=args.pool,
                            dense_unit=args.dense,
                            lambda_regularization=None)
        
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True,min_delta=0.01)
        
        
    #fitting the model 
    history = model.fit(x=dat_out['ins_train'],y=dat_out['outs_train'],batch_size=32,epochs=15,validation_data=(dat_out['ins_valid'],dat_out['outs_valid']),callbacks=[early_stopping_cb])
        
        
        
    #model evaluate
    evalu = model.evaluate(dat_out['ins_test'], dat_out['outs_test'])
    
    #model summary
    print(model.summary())
   
    
    
    #plot model
    plot_model(model, to_file='model_plot_%d.png'%args.rotation, show_shapes=True, show_layer_names=True)
    
    #dumping the outputs into pickel files
    fname1 = 'historyn_%d.pkl'%args.rotation
    fname2 = 'evaluationn_%d.pkl'%args.rotation
    
    with open(fname1, 'wb') as fp:
        pickle.dump(history.history, fp)
    
    with open(fname2, 'wb') as fp1:
        pickle.dump(evalu, fp1)
    
