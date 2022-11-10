
# Standard libraries
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential

# NOT PROVIDED (but has lots of relevant TF imports)
#from deep_networks import *

# PROVIDED
from symbiotic_metrics import *
from job_control import *



#################################################################
def deep_network_basic(n_inputs, n_hidden, n_output, activation, lrate, dropout, dropout_input, kernel_regularizer, metrics):
    
    if dropout is not None:
        print("Dense: DROPOUT %f"%dropout)
    if kernel_regularizer is not None:
        print("Dense: L2_regularization %f"%kernel_regularizer)
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
    
    #sequential model
    model = Sequential();
    #input
    model.add(InputLayer(input_shape = (n_inputs,)))
    
    #dropout_input
    if dropout_input is not None:
        model.add(Dropout(rate = dropout_input, name = "dropout_input"))
    
    #hidden layers
    model.add(Dense(n_hidden[0], use_bias = True, name = "hidden_1", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_1"))
    model.add(Dense(n_hidden[1], use_bias = True, name = "hidden_2", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_2"))
    model.add(Dense(n_hidden[2], use_bias = True, name = "hidden_3", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_3"))
    model.add(Dense(n_hidden[3], use_bias = True, name = "hidden_4", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_4"))
    model.add(Dense(n_hidden[4], use_bias = True, name = "hidden_5", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_5"))
    model.add(Dense(n_hidden[5], use_bias = True, name = "hidden_6", activation=activation, kernel_regularizer = kernel_regularizer))
    if dropout is not None:
        model.add(Dropout(rate = dropout, name = "dropout_6"))
    
    #output
    model.add(Dense(n_output, use_bias = True, name = "output", activation='linear'))
    
    #optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    #compiling the model
    model.compile(loss='mse', optimizer=opt, metrics = metrics)
    
    return model

#################################################################
def extract_data(bmi, args):
    '''
    Translate BMI data structure into a data set for training/evaluating a single model
    
    :param bmi: Dictionary containing the full BMI data set, as loaded from the pickle file.
    :param args: Argparse object, which contains key information, including Nfolds, 
            predict_dim, output_type, rotation
            
    :return: Numpy arrays in standard TF format for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of folds that have been chosen
    '''
    # Number of folds in the data set
    ins = bmi['MI']
    Nfolds = len(ins)
    times = bmi['time']
    
    # Check that argument matches actual number of folds
    assert (Nfolds == args.Nfolds), "Nfolds must match folds in data set"
    
    # Pull out the data to be predicted
    outs = bmi[args.output_type]
    
    # Check that predict_dim is valid
    assert (args.predict_dim is None or (args.predict_dim >= 0 and args.predict_dim < outs[0].shape[1]))
    
    # Rotation and number of folds to use for training
    r = args.rotation
    Ntraining = args.Ntraining
    
    # Compute which folds belong in which set
    folds_training = (np.array(range(Ntraining)) + r) % Nfolds
    folds_validation = (np.array([Nfolds-2]) +r ) % Nfolds
    folds_testing = (np.array([Nfolds-1]) + r) % Nfolds
    
    # Log these choices
    folds = {'folds_training': folds_training, 'folds_validation': folds_validation,
            'folds_testing': folds_testing}
    
    # Combine the folds into training/val/test data sets (pairs of input/output numpy arrays)
    ins_training = np.concatenate(np.take(ins, folds_training))
    outs_training = np.concatenate(np.take(outs, folds_training))
    time_training = np.concatenate(np.take(times, folds_training))
        
    ins_validation = np.concatenate(np.take(ins, folds_validation))
    outs_validation = np.concatenate(np.take(outs, folds_validation))
    time_validation = np.concatenate(np.take(times, folds_validation))
        
    ins_testing = np.concatenate(np.take(ins, folds_testing))
    outs_testing = np.concatenate(np.take(outs, folds_testing))
    time_testing = np.concatenate(np.take(times, folds_testing))
    
    # If a particular dimension is specified, then extract it from the outputs
    if args.predict_dim is not None:
        outs_training = outs_training[:,[args.predict_dim]]
        outs_validation = outs_validation[:,[args.predict_dim]]
        outs_testing = outs_testing[:,[args.predict_dim]]
    
    return ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, folds, time_training, time_validation, time_testing

def augment_args(args):
    '''
    Use the jobiterator to override the command-line arguments based on the experiment index. 

    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index == -1):
        return "Ntraining_%d_rotation_%d"%(args.Ntraining, args.rotation)
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = {'Ntraining': [1, 2, 3, 5, 8, 12, 18], 
         'rotation': range(20),
        'dropout': [0.1, 0.25, 0.4]}

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
    
def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    
    Expand this as needed
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Dimension being predicted
    if args.predict_dim is None:
        predict_str = args.output_type
    else:
        predict_str = '%s_%d'%(args.output_type, args.predict_dim)
        
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/%s_%s_hidden_%s_%s"%(args.results_path, args.exp_type, predict_str, hidden_str, params_str)

def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args in specific situations
    params_str = augment_args(args)
    
    print("Params:", params_str)
    
    # Compute output file name base
    fbase = generate_fname(args, params_str)
    
    print("File name base:", fbase)
    
    # Is this a test run?
    if(args.nogo):
        # Don't execute the experiment
        print("Test run only")
        return None
    
    # Load the data
    fp = open(args.dataset, "rb")
    bmi = pickle.load(fp)
    fp.close()
    
    # Extract the data sets.  This process uses rotation and Ntraining (among other exp args)
    ins, outs, ins_validation, outs_validation, ins_testing, outs_testing, folds, time_training, time_validation, time_testing = extract_data(bmi, args)
    
    # Metrics
    fvaf = FractionOfVarianceAccountedFor(outs.shape[1])
    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Build the model: you are responsible for providing this function
    model = deep_network_basic(ins.shape[1], tuple(args.hidden), outs.shape[1],     # Size of inputs, hidden layer(s) and outputs
                               activation='elu',
                              lrate=args.lrate,
                              dropout = args.dropout, 
                              dropout_input = args.dropout,
                              kernel_regularizer = args.L2_regularization,
                              metrics=[fvaf, rmse])
    
    # Report if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)
    
    # Learn
    history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=args.verbose>=2,
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=[early_stopping_cb])
        
    # Generate log data
    results = {}
    results['args'] = args
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['time_training'] = time_training
    results['predict_validation'] = model.predict(ins_validation)
    results['predict_validation_eval'] = model.evaluate(ins_validation, outs_validation)
    results['time_validation'] = time_validation
    results['predict_testing'] = model.predict(ins_testing)
    results['predict_testing_eval'] = model.evaluate(ins_testing, outs_testing)
    results['time_testing'] = time_testing
    results['folds'] = folds
    results['history'] = history.history
    
    # Save results
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Save the model (can't be included in the pickle file)
    model.save("%s_model"%(fbase))
    return model
               
def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/bmi/bmi_dataset.pkl', help='Data set file')
    parser.add_argument('--Ntraining', type=int, default=2, help='Number of training folds')
    parser.add_argument('--output_type', type=str, default='torque', help='Type to predict')
    parser.add_argument('--exp_index', type=int, default=-1, help='Experiment index')
    parser.add_argument('--Nfolds', type=int, default=20, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--hidden', nargs='+', type=int, default=[200, 100, 50, 25, 10, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--predict_dim', type=int, default=1, help="Dimension of the output to predict")
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--exp_type', type=str, default='bmi', help='High level name for this set of experiments')
    parser.add_argument('--dropout', type=float, default=None, help="Dropout rate")
    parser.add_argument('--L1_regularization','--l1', type=float, default=None, help="L1 regularization factor")
    parser.add_argument('--L2_regularization','--l2', type=float, default=None, help="L2 regularization factor")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    
    return parser

def check_args(args):
    '''
    Check that key arguments are within appropriate bounds.  Failing an assert causes a hard failure with meaningful output
    '''
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-2)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
     #Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
     #GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    execute_exp(args)