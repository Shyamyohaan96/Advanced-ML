'''

Image classification

'''

import sys
import argparse
import pickle
import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,InputLayer
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.models import Sequential
#############
# REMOVE THESE LINES if symbiotic_metrics, job_control, networks are in the same directory
#tf_tools = "../../../../tf_tools/"
#sys.path.append(tf_tools + "metrics")
#sys.path.append(tf_tools + "experiment_control")
#sys.path.append(tf_tools + "networks")
#############

# Provided
from symbiotic_metrics import *
from job_control import *

# You need to provide this yourself
#from cnn_classifier import *

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

def create_cnn_classifier_network(image_size, nchannels, conv_layers, dense_layers, p_dropout, lambda_l2, lrate,
                                  n_classes):
    if lambda_l2 is not None:
        lambda_l2 = tf.keras.regularizers.l2(lambda_l2)
    
    #sequential model
    model = Sequential();
    #input
    model.add(InputLayer(input_shape =(image_size[0], image_size[1], nchannels)))
    # convolution layer
    for i in range(len(conv_layers)):
        model.add(Conv2D(conv_layers[i]['filters'], 
                         conv_layers[i]['kernel_size'], 
                         strides = 1,
                         padding = 'valid',
                         use_bias = True,
                         kernel_initializer = 'random_uniform',
                         bias_initializer = 'zeros',
                         name = 'Convolution_'+str(i+1),
                         activation = 'elu',
                         kernel_regularizer = lambda_l2))
        #Pooling layer
        if conv_layers[i]['pool_size'] is not None:
            model.add(MaxPooling2D(pool_size =conv_layers[i]['pool_size'],
                                   strides = conv_layers[i]['strides']))
    #Flatenning layer    
    model.add(Flatten())
    
    #Dense layers
    for i in range(len(dense_layers)):
        model.add(Dense(units = dense_layers[i]['units'],
                        use_bias = True,
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'zeros',
                        name = 'Dense_'+str(i+1),
                        activation = 'elu',
                        kernel_regularizer = lambda_l2))
        model.add(Dropout(p_dropout))
    
    #output layer
    model.add(Dense(units = n_classes ,
                    kernel_initializer = 'truncated_normal',
                    bias_initializer = 'zeros',
                    name = 'output',
                    activation = 'softmax',
                    kernel_regularizer = lambda_l2))
    
    #optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                   amsgrad=False)
    
    #compiling the model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['categorical_accuracy'])
    
    print(model.summary())
    
    return model
        

def load_meta_data_set(fname = 'classes.pkl'):
    '''
    Load the metadata for a multi-class data set

    The returned data set is:
    - A python list of classes
    - Each class has a python list of object
    - Each object is described as using a DataFrame that contains file
       locations/names and class labels


    :param fname: Pickle file with the metadata

    :return: Dictionary containing class data
    '''

    obj = None
    
    with open(fname, "rb") as fp:
        obj = pickle.load(fp)

    assert obj is not None, "Meta-data loading error"
    
    print("Read %d classes\n"%obj['nclasses'])
    
    return obj


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files");
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/core50/core50_128x128', help='Data set directory')
    parser.add_argument('--image_size', nargs=3, type=int, default=[128,128,3], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--classes', type=str, default='classes.pkl', help='Name of file containing the classes metadata')
    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Convolutional parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--L1_regularizer', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularizer', '--l2', type=float, default=None, help="L2 regularization parameter")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--steps_per_epoch', type=int, default=10, help="Number of gradient descent steps per epoch")
    parser.add_argument('--validation_fraction', type=float, default=0.1, help="Fraction of available validation set to actually use for validation")
    parser.add_argument('--testing_fraction', type=float, default=0.5, help="Fraction of available testing set to actually use for testing")
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    
    return parser


def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        p = {'rotation': range(5)}
    else:
        assert False, "Unrecognized exp_type"

    return p


#################################################################
def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularizer is None or (args.L1_regularizer > 0.0 and args.L1_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    assert (args.L2_regularizer is None or (args.L2_regularizer > 0.0 and args.L2_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if(index is None):
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Conv configuration
    conv_size_str = '_'.join(str(x) for x in args.conv_size)
    conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)
    pool_str = '_'.join(str(x) for x in args.pool)
    
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout)
        
    # L1 regularization
    if args.L1_regularizer is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_'%(args.L1_regularizer)

    # L2 regularization
    if args.L2_regularizer is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_regularizer)


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
        
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f_"%args.lrate
    
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/image_%s%sCsize_%s_Cfilters_%s_Pool_%s_hidden_%s_%s%s%s%sntrain_%02d_rot_%02d"%(args.results_path,
                                                                                           experiment_type_str,
                                                                                           label_str,
                                                                                           conv_size_str,
                                                                                           conv_filter_str,
                                                                                           pool_str,
                                                                                           hidden_str, 
                                                                                           dropout_str,
                                                                                           regularizer_l1_str,
                                                                                           regularizer_l2_str,
                                                                                           lrate_str,
                                                                                           args.Ntraining,
                                                                                           args.rotation)


#################################################################
def split_dataframes(rotation, classes, ntraining):
    '''
    Translate the metadata set into DataFrames that represent the training,
    validation and testing sets, respectively

    :param rotation: Integer rotation (0 <= rot < nobjects)
    :param classes: metadata set
    :param ntraining: Number of training objects.
            If ntraining <= Nobjects-2, then we will have both validation and test sets
            If ntraining == Nobjects-1, then we will have only a validation set 
    
    '''
    # We assume that each class has the same number of objects
    cl = classes['classes']
    nobjects = len(cl[0])

    assert (ntraining < nobjects), "Ntraining must be less than number of objects"

    if ntraining == nobjects - 1:
        # Object indices for training and validation sets
        objects_training = (np.arange(ntraining) + rotation) % nobjects
        objects_validation = (nobjects - 1 + rotation ) % nobjects
        objects_testing = None
        df_testing = None
    else:
        # Object indices for training, validation and test sets
        objects_training = (np.arange(ntraining) + rotation) % nobjects
        objects_validation = (nobjects - 2 + rotation ) % nobjects
        objects_testing = (nobjects - 1 + rotation ) % nobjects
        df_testing = pd.DataFrame(columns=['x_col', 'y_col'])

    print("Training: ", objects_training)
    print("Validation: ", objects_validation)
    print("Testing: ", objects_testing)
    
    # Initially empty dataframes for all data sets
    df_training = pd.DataFrame(columns=['x_col', 'y_col'])
    df_validation = pd.DataFrame(columns=['x_col', 'y_col'])

    # Iterate over classes
    for c in cl:
        # Iterate over objects in training set
        for i in objects_training:
            # Append new object to training set
            df_training = pd.concat([df_training, c[i]])

        # Append new object to validation set
        df_validation = pd.concat([df_validation, c[objects_validation]])

        # If there is also a test set, then append objects to it
        if objects_testing is not None:
            df_testing = pd.concat([df_testing, c[objects_testing]])

    # Done
    return df_training, df_validation, df_testing
        

#################################################################
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    '''
    
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)
    
    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    
    # Load meta data
    classes = load_meta_data_set(args.classes)

    # Split metadata into individual data sets
    df_train, df_validation, df_testing = split_dataframes(args.rotation, classes, args.Ntraining)

    # Compute the number of samples in each data set
    nsamples_train = df_train.size/2
    nsamples_validation = df_validation.size/2
    if df_testing is None:
        nsamples_testing = 0
    else:
        nsamples_testing = df_testing.size/2

    print("Total samples: Tr:%d, V:%d, Te:%d"%(nsamples_train, nsamples_validation, nsamples_testing))

    ####################################################3
    # Generator for training set

    # NOTE: you may wish to add data augmentation here
    train_generator=ImageDataGenerator(rescale=1./255.)

    # Labels are string category labels (valid for categorical class mode)
    # df_train contains file names that are relative to args.dataset
    # shuffle=True -> shuffle before grouping into batches
    train_flow = train_generator.flow_from_dataframe(dataframe=df_train,
                                                     directory=args.dataset,
                                                     x_col="file",
                                                     y_col="label",
                                                     batch_size=args.batch,
                                                     seed=args.generator_seed,
                                                     shuffle=True,
                                                     class_mode="categorical",
                                                     target_size=args.image_size[0:2])

    # Generator for validation set.  We will only take a small subset of the
    #  available validation set
    # This generator is separate from above because we partitioned our train/validation split by hand

    # NOTE: NEVER ADD DATA AUGMENTATION TO THE VALIDATION OR TEST SETS
    validation_generator=ImageDataGenerator(rescale=1./255.,
                                            validation_split=args.validation_fraction)

    # By selecting subset=validation, the true dataset is only the fraction that is specified above
    validation_flow = validation_generator.flow_from_dataframe(dataframe=df_validation,
                                                               directory=args.dataset,
                                                               x_col="file",
                                                               y_col="label",
                                                               batch_size=args.batch,
                                                               seed=args.generator_seed,
                                                               shuffle=False,
                                                               class_mode="categorical",
                                                               subset="validation",
                                                               target_size=args.image_size[0:2])

    ####################################################
    # Build the model
    image_size=args.image_size[0:2]
    nchannels = args.image_size[2]

    # Network config
    # NOTE: this is very specific to our implementation of create_cnn_classifier_network()
    #   List comprehension and zip all in one place (ugly, but effective).
    #   Feel free to organize this differently
    dense_layers = [{'units': i} for i in args.hidden]
    conv_layers = [{'filters': f, 'kernel_size': (s,s), 'pool_size': (p,p), 'strides': (p,p)} if p > 1
                   else {'filters': f, 'kernel_size': (s,s), 'pool_size': None, 'strides': None}
                   for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.pool)]
    
    print("Dense layers:", dense_layers)
    print("Conv layers:", conv_layers)
    
    # Build network: you must provide your own implementation
    model = create_cnn_classifier_network(image_size,
                                          nchannels,
                                          conv_layers=conv_layers,
                                          dense_layers=dense_layers,
                                          p_dropout=args.dropout,
                                          lambda_l2=args.L2_regularizer,
                                          lrate=args.lrate, n_classes=classes['nclasses'])
    
    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl"%fbase
    
    # Perform the experiment?
    if(args.nogo):
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if os.path.exists(fname_out):
            # Results file does exist: exit
            print("File %s already exists"%fname_out)
            return
            
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #  validation_steps=None means that ALL validation samples will be used (of the selected subset)
    history = model.fit(train_flow,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        use_multiprocessing=False, 
                        verbose=args.verbose>=2,
                        validation_data=validation_flow,
                        validation_steps=None, 
                        callbacks=[early_stopping_cb])

    # Create generator for test set
    if df_testing is not None:
        # Generator for testing set.  We will only take a small subset of the
        #  available testing set
        # This generator is separate from above because we partitioned our train/testing split by hand
        testing_generator=ImageDataGenerator(rescale=1./255.,
                                             validation_split=args.testing_fraction)

        # By selecting subset=validation, the true dataset is only the fraction that is specified above
        testing_flow = testing_generator.flow_from_dataframe(dataframe=df_testing,
                                                             directory=args.dataset,
                                                             x_col="file",
                                                             y_col="label",
                                                             batch_size=args.batch,
                                                             seed=args.generator_seed,
                                                             shuffle=False,
                                                             class_mode="categorical",
                                                             subset="validation",
                                                             target_size=args.image_size[0:2])
    


    # Generate results data
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(validation_flow)
    results['predict_validation_eval'] = model.evaluate(validation_flow)
    
    if df_testing is not None:
        results['predict_testing'] = model.predict(testing_flow)
        results['predict_testing_eval'] = model.evaluate(testing_flow)
        
    results['predict_training'] = model.predict(train_flow)
    results['predict_training_eval'] = model.evaluate(train_flow)
    results['history'] = history.history

    # TODO: really want to know the order of the objects in the validation and testing dataset
    
    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    model.save("%s_model"%(fbase))

    print(fbase)
    
    return model


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    if(args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment
        execute_exp(args)
        