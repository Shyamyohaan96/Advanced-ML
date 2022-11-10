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

for i in range(5):
    #loading the test dataset
    dataset_test = chesapeake_loader.create_dataset(base_dir='/home/fagg/datasets/radiant_earth/pa',
                                               partition='valid', fold=i, filt='*', 
                                               batch_size=32, prefetch=2, num_parallel_calls=4)
    
    #loading the saved model
    model = keras.models.load_model('model_%d'%i)
    
    #initializing the list
    test_feature = []
    test_label = []
    test_prediction = []
    test_prediction_label = []
    
    #prediction of the model
    for j in dataset_test:
        test_feature.append(j[0].numpy())
        test_label.append(j[1].numpy())
        pred = model.predict(j[0].numpy())
        test_prediction.append(pred)
        test_prediction_label.append(np.argmax(pred,axis=3))
        
    
    #bringing all values from different batches together
    test_feature = np.concatenate(test_feature, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    test_prediction = np.concatenate(test_prediction, axis=0)
    test_prediction_label = np.concatenate(test_prediction_label, axis=0)
    
    
    #storing values in dictionary
    test = {}
    test['test_feature'] = test_feature
    test['test_label'] = test_label
    test['test_prediction'] = test_prediction
    test['test_prediction_label'] = test_prediction_label
    
    #dumping the outputs into pickel files
    fname1 = 'prediction_%d.pkl'%i
    
    with open(fname1, 'wb') as fp:
        pickle.dump(test, fp)
    
    
            
    