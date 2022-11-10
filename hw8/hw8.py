import sys
import argparse
import pickle
import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Input, concatenate, SpatialDropout2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

import re
import chesapeake_loader
import argparse

##################

def semantic_model(spa_accuracy,  activation_dense, con_filter, con_kernel, con_pool, trans_filter, lrate):
    
    #initializing the sequential model
    model = Sequential()
    
    #input layer
    model.add(Input(shape =(256, 256, 26)))
    
    #building convolution and maxpooling layers
    for i in range(len(con_filter)):
        model.add(Conv2D(con_filter[i], 
                        (con_kernel[i],con_kernel[i]),
                        padding = 'same',
                        name = 'Convolution_'+str(i+1),
                        activation=activation_dense))
        
        model.add(MaxPooling2D(pool_size=(con_pool[i],con_pool[i])))
    
    # building transpose convolution layers
    for i in range(len(trans_filter)):
        model.add(Conv2DTranspose(trans_filter[i], 
                        (con_kernel[i],con_kernel[i]),
                        strides = (2, 2), 
                        padding = 'same',
                        name = 'transconvolution_'+str(i+1),
                        activation=activation_dense))
    
    #output layer
    model.add(Conv2D(7, 
                    (3, 3) , 
                    padding='same',
                    name = 'output',
                    activation='softmax'))
    
    #The optimizer determines how the gradient descent is to be done
    opt = keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
    
    #compiling the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=[spa_accuracy])
    
    return model    
    
    

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='semantic', fromfile_prefix_chars='@')
    
    # High-level experiment configuration
    parser.add_argument('--name', type=str, default=None, help="name of the model")
    parser.add_argument('--rotation', type=int, default=0, help='rotation')
    
    #parameters
    parser.add_argument('--lrate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.001, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early termination")
    
    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--steps_per_epoch', type=int, default=10, help="Number of gradient descent steps per epoch")

    
    #parameters for building model
    parser.add_argument('--conv_kernel', nargs='+', type=int, default=[3,5], help='Convolution kernel size size per layer')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--transpose_con', nargs='+', type=int, default=[15, 10], help='Number of transpose Convolution filters per layer')
        
    return parser

if __name__ == "__main__":
    
    #initializing the parser
    parser = create_parser()
    args = parser.parse_args()
    
    #initializing the training set
    dataset_train = chesapeake_loader.create_dataset(base_dir='/home/fagg/datasets/radiant_earth/pa',
                                               partition='train', fold=args.rotation, filt='*[012345678]', 
                                               batch_size=args.batch, prefetch=2, num_parallel_calls=4)
    
    #initializing the validation set
    dataset_valid = chesapeake_loader.create_dataset(base_dir='/home/fagg/datasets/radiant_earth/pa',
                                               partition='train', fold=args.rotation, filt='*[9]', 
                                               batch_size=args.batch, prefetch=2, num_parallel_calls=4)
    
    
    # intializing the accuracy metric
    spa_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    #building the model
    model = semantic_model(spa_accuracy,  
                           activation_dense='elu',
                           con_filter=args.conv_nfilters,
                           con_kernel=args.conv_kernel,
                           con_pool=args.pool,
                           trans_filter=args.transpose_con,
                           lrate = args.lrate)
    
    #plot model
    plot_model(model, to_file='model_plot_%d.png'%args.rotation, show_shapes=True, show_layer_names=True)
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)
    
    #fitting the model 
    history = model.fit(x= dataset_train,
                        epochs=args.epochs,
                        validation_data= dataset_valid,
                        callbacks=[early_stopping_cb])
    
    #model summary
    print(model.summary())
    
    #saving the model
    model.save('model_%d'%args.rotation)
    
    
    #dumping the outputs into pickel files
    fname1 = 'history_%d.pkl'%args.rotation
    
    with open(fname1, 'wb') as fp:
        pickle.dump(history.history, fp)
    
    
    


      