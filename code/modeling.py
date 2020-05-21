
#!usr/bin/env python3

'''
30 day Readmission after ICU stay using MIMIC III
modified from original paper repo: https://github.com/mit-ddig/multitask-patients

@author: Sparkle Russell-Puleri
@date: May, 9th 2020

'''

import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import sys
from datetime import datetime as dt
import json
from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(1)
import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam 
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import pickle
from sklearn import metrics
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K
from preprocessing import * 



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='readmission_test',
                        help="This will become the name of the folder where are the models and results \
        are stored. Type: String. Default: 'readmission_test'.")
    parser.add_argument("--model_type", type=str, default='SEPARATE',
                        help="One of {'GLOBAL', MULTITASK', 'SEPARATE'} indicating \
        which type of model to run. Type: String.")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
                        help="Number of beginning LSTM layers, applies to all model types. \
        Type: int. Default: 1.")
    parser.add_argument("--lstm_layer_size", type=int, default=8,
                        help="Number of units in beginning LSTM layers, applies to all model types. \
        Type: int. Default: 16.")
    parser.add_argument("--num_dense_shared_layers", type=int, default=0,
                        help="Number of shared dense layers following LSTM layer(s), applies to \
        all model types. Type: int. Default: 0.")
    parser.add_argument("--dense_shared_layer_size", type=int, default=0,
                        help="Number of units in shared dense layers, applies to all model types. \
        Type: int. Default: 0.")
    parser.add_argument("--num_multi_layers", type=int, default=0,
                        help="Number of separate-task dense layers, only applies to multitask models. Currently \
        only 0 or 1 separate-task dense layers are supported. Type: int. Default: 0.")
    parser.add_argument("--multi_layer_size", type=int, default=0,
                        help="Number of units in separate-task dense layers, only applies to multitask \
        models. Type: int. Default: 0.")
    parser.add_argument("--cohorts", type=str, default='custom',
                        help="One of {'careunit', 'custom'}. Indicates whether to use pre-defined cohorts \
        (careunits ) or use a custom cohort membership (i.e. result of clustering). \
        Type: String. Default: 'careunit'. ")
    parser.add_argument("--cohort_filepath", type=str, 
                        help="This is the filename containing a numpy \
                              array of length len(X), containing the cohort membership for each example\
                              in X. This file should be saved in the folder 'cluster_membership'\
                              . Only applies to cohorts == 'custom'. Type: str.", 
                        default='test_clusters.npy')
    parser.add_argument("--sample_weights", action="store_true", default=False, help="This is an indicator \
        flag to weight samples during training by their cohort's inverse frequency (i.e. smaller cohorts will be \
        more highly weighted during training).")
    parser.add_argument("--include_cohort_as_feature", action="store_true", default=False,
                        help="This is an indicator flag to include cohort membership as an additional feature in the matrix.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train for. Type: int. Default: 30.")
    parser.add_argument("--train_val_random_seed", type=int, default=0,
                        help="Random seed to use during train / val / split process. Type: int. Default: 0.")
    parser.add_argument("--repeats_allowed", action="store_true", default=True,
                        help="Indicator flag allowing training and evaluating of existing models. Without this flag, \
        if you run a configuration for which you've already saved models & results, it will be skipped.")
    parser.add_argument("--no_val_bootstrap", action="store_true", default=False,
                        help="Indicator flag turning off bootstrapping evaluation on the validation set. Without this flag, \
        minimum, maximum and average AUCs on bootstrapped samples of the validation dataset are saved. With the flag, \
        just one AUC on the actual validation set is saved.")
    parser.add_argument("--num_val_bootstrap_samples", type=int, default=100,
                        help="Number of bootstrapping samples to evaluate on for the validation set. Type: int. Default: 100. ")
    parser.add_argument("--test_time", action="store_true", default=False,
                        help="Indicator flag of whether we are in testing time. With this flag, we will load in the already trained model \
        of the specified configuration, and evaluate it on the test set. ")
    parser.add_argument("--test_bootstrap", action="store_true", default=False,
                        help="Indicator flag of whether to evaluate on bootstrapped samples of the test set, or just the single \
        test set. Adding the flag will result in saving minimum, maximum and average AUCs on bo6otstrapped samples of the validation dataset. ")
    parser.add_argument("--num_test_bootstrap_samples", type=int, default=100,
                        help="Number of bootstrapping samples to evaluate on for the test set. Type: int. Default: 100. ")
    parser.add_argument('--data_path',  type=str, help='Path to save physiological data',  default='data/')
    parser.add_argument('--save_data_path',  type=str, help='Path to save outputs',  default='data/')
    parser.add_argument('--db_config', '-db', type=str, help='File path to the databse JSON file',
                        default='../db_login.JSON')

    args = parser.parse_args()
    print(args)
    return args

################ CREATE MODELS ###############################################
####################################################################################

def create_single_task_model(n_layers, units, num_dense_shared_layers, dense_shared_layer_size, input_dim, output_dim):
    """ 
    Create a single task model with LSTM layer(s), shared dense layer(s), and sigmoided output. 
    Args:
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        input_dim (int): Number of features in the input.
        output_dim (int): Number of outputs (1 for binary tasks).
    Returns: 
        final_model (Keras model): A compiled model with the provided architecture. 
    """

    # global model
    model = tf.keras.Sequential()

    # first layer
    if n_layers > 1:
        return_seq = True
    else:
        return_seq = False

    model.add(tf.keras.layers.LSTM(units=units, activation='relu',
                   input_shape=input_dim, return_sequences=return_seq,
                                   kernel_initializer='uniform'))

    # additional hidden layers
    for l in range(n_layers - 1):
        model.add(tf.keras.layers.LSTM(units=units, activation='relu'))

    # additional dense layers
    for l in range(num_dense_shared_layers):
        model.add(tf.keras.layers.Dense(units=dense_shared_layer_size, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    

    return model

def create_multitask_model(input_dim, n_layers, units, num_dense_shared_layers, dense_shared_layer_size, n_multi_layers, multi_units, output_dim, tasks):
    """ 
    Create a multitask model with LSTM layer(s), shared dense layer(s), separate dense layer(s) 
    and separate sigmoided outputs. 
    Args: 
        input_dim (int): Number of features in the input.
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        n_multi_layers (int): Number of task-specific dense layers. 
        multi_layer_size (int): Number of units in each task-specific dense layer.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
    Returns: 
        final_model (Keras model): A compiled model with the provided architecture. 
    """

    tasks = [str(t) for t in tasks]
    n_tasks = len(tasks)

    # Input layer
    x_inputs = tf.keras.Input(shape=input_dim)

    # first layer
    if n_layers > 1:
        return_seq = True
    else:
        return_seq = False

    # Shared layers
    combined_model = tf.keras.layers.LSTM(units, activation='relu',
                          input_shape=input_dim,
                          name='combined', return_sequences=return_seq)(x_inputs)

    for l in range(n_layers - 1):
        combined_model = tf.keras.layers.LSTM(units, activation='relu')(combined_model)

    for l in range(num_dense_shared_layers):
        combined_model = tf.keras.layers.Dense(dense_shared_layer_size,
                               activation='relu')(combined_model)

    # Individual task layers
    if n_multi_layers == 0:
        # Only create task-specific output layer.
        output_layers = []
        for task_num in range(n_tasks):
            output_layers.append(tf.keras.layers.Dense(output_dim, activation='sigmoid',
                                       name=tasks[task_num])(combined_model))

    else:
        # Also create task-specific dense layer.
        task_layers = []
        for task_num in range(n_tasks):
            task_layers.append(tf.keras.layers.Dense(multi_units, activation='relu',
                                     name=tasks[task_num])(combined_model))

        output_layers = []
        for task_layer_num in range(len(task_layers)):
            output_layers.append(tf.keras.layers.Dense(output_dim, activation='sigmoid',
                                       name=str(tasks[task_layer_num]) + '_output')(task_layers[task_layer_num]))

    loss_fn = 'binary_crossentropy'
    learning_rate = 0.0001
    final_model = Model(inputs=x_inputs, outputs=output_layers)
    final_model.compile(loss=loss_fn,
                        optimizer=tf.keras.optimizers.Adam(learning_rate),
                        metrics=['accuracy'])

    return final_model

################ RUNNING MODELS ###############################################
####################################################################################


def run_separate_models(X_train, y_train, cohorts_train,
                        X_val, y_val, cohorts_val,
                        X_test, y_test, cohorts_test,
                        all_tasks, fname_keys, fname_results,
                        FLAGS):
    """
    Train and evaluate separate models for each task. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_separate_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    cohort_aucs = []

    # if we're testing, just load the model and save results
    if FLAGS.test_time:
        for task in all_tasks:
            model_fname_parts = ['separate', str(task), 
                                 'lstm_shared', str(FLAGS.num_lstm_layers), 
                                 'layers', str(FLAGS.lstm_layer_size), 'units',
                                 str(FLAGS.num_dense_shared_layers), 
                                 'dense_shared', str(FLAGS.dense_shared_layer_size), 
                                 'dense_units', 'mortality']
            
            model_path = FLAGS.experiment_name + \
                '/models/' + "_".join(model_fname_parts)
            model = load_model(model_path)

            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test, task, model, return_everything=False,
                                             test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))

            else:
                x_test_in_task = X_test[cohorts_test == task]
                y_test_in_task = y_test[cohorts_test == task]

                y_pred = model.predict(x_test_in_task)
                auc = roc_auc_score(y_test_in_task, y_pred)
                cohort_aucs.append(auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_separate_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    # otherwise, create and train a model
    for task in all_tasks:

        # get training data from cohort
        x_train_in_task = X_train[cohorts_train == task]
        y_train_in_task = y_train[cohorts_train == task]

        x_val_in_task = X_val[cohorts_val == task]
        y_val_in_task = y_val[cohorts_val == task]

        # create & fit model
        model = create_single_task_model(FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                                         FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size, X_train.shape[1:], 1)
        print(model.summary)
        model_fname_parts = ['separate', str(task), 'lstm_shared', 
                             str(FLAGS.num_lstm_layers), 'layers', 
                             str(FLAGS.lstm_layer_size), 'units',
                             str(FLAGS.num_dense_shared_layers), 
                             'dense_shared', 
                             str(FLAGS.dense_shared_layer_size), 
                             'dense_units', 'readmission']
        
        model_dir = FLAGS.experiment_name + \
            '/checkpoints/' + "_".join(model_fname_parts)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            model_fname, monitor='val_loss', verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        model.fit(x_train_in_task, y_train_in_task, epochs=FLAGS.epochs, batch_size=100,
                  callbacks=[checkpointer, early_stopping],
                  validation_data=(x_val_in_task, y_val_in_task))

        # make validation predictions & evaluate
        preds_for_cohort = model.predict(x_val_in_task, batch_size=128)

        print('AUC of separate model for ', task, ':')
        if FLAGS.no_val_bootstrap:
            try:
                auc = roc_auc_score(y_val_in_task, preds_for_cohort)
            except:
                auc = np.nan

            cohort_aucs.append(auc)
        else:
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, model, 
                return_everything=False, 
                num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            auc = avg_auc
            print("(min/max/average):")

        print(cohort_aucs[-1])
       

        model.save(FLAGS.experiment_name + '/models/' +
                   "_".join(model_fname_parts))

    # save results to a file
    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                          FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
    try:
        separate_model_results = np.load(fname_results)
        separate_model_key = np.load(fname_keys)
        separate_model_results = np.concatenate(
            (separate_model_results, np.expand_dims(cohort_aucs, 0)))
        separate_model_key = np.concatenate(
            (separate_model_key, np.array([current_run_params])))

    except:
        separate_model_results = np.expand_dims(cohort_aucs, 0)
        separate_model_key = np.array([current_run_params])

    np.save(fname_results, separate_model_results)
    np.save(fname_keys, separate_model_key)
    print('Saved separate results.')

def run_multitask_model(X_train, y_train, cohorts_train,
                        X_val, y_val, cohorts_val,
                        X_test, y_test, cohorts_test,
                        all_tasks, fname_keys, fname_results,
                        FLAGS):
    """
    Train and evaluate multitask model. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_separate_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    model_fname_parts = ['mtl', 'lstm_shared', str(FLAGS.num_lstm_layers), 
                         'layers', str(FLAGS.lstm_layer_size), 'units',
                         'dense_shared', str(FLAGS.num_dense_shared_layers), 
                         'layers', str(FLAGS.dense_shared_layer_size), 'dense_units',
                         'specific', str(FLAGS.num_multi_layers), 'layers', 
                         str(FLAGS.multi_layer_size), 
                         'specific_units', 'readmission']

    n_tasks = len(np.unique(cohorts_train))
    cohort_key = dict(zip(all_tasks, range(n_tasks)))

    if FLAGS.test_time:
        model_path = FLAGS.experiment_name + \
            '/models/' + "_".join(model_fname_parts)
        model = load_model(model_path)
        y_pred = model.predict(X_test)
        
        cohort_aucs = []
        for task in all_tasks:
            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test,
                                             task=task, model=model, return_everything=True, test=True,
                                             all_tasks=all_tasks,
                                             num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))
            else:
                y_pred_in_cohort = y_pred[cohorts_test ==
                                          task, cohort_key[task]]
                y_true_in_cohort = y_test[cohorts_test == task]
                auc = roc_auc_score(y_true_in_cohort, y_pred_in_cohort)
                cohort_aucs.append(auc)

        if FLAGS.test_bootstrap:
            cohort_aucs = np.array(cohort_aucs)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.expand_dims(np.mean(cohort_aucs, axis=0), 0)))

            all_micro_aucs = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model, return_everything=True, test=True,
                                               all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.array([all_micro_aucs])))

        else:
            macro_auc = np.mean(cohort_aucs)
            cohort_aucs.append(macro_auc)
            micro_auc = roc_auc_score(y_test, y_pred[np.arange(len(y_test)), [
                                      cohort_key[c] for c in cohorts_test]])
            cohort_aucs.append(micro_auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_multitask_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    # model
    mtl_model = create_multitask_model(X_train.shape[1:], FLAGS.num_lstm_layers,
                                       FLAGS.lstm_layer_size, FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size,
                                       FLAGS.num_multi_layers, FLAGS.multi_layer_size, output_dim=1, tasks=all_tasks)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

    model_dir = FLAGS.experiment_name + \
        '/checkpoints/' + "_".join(model_fname_parts)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_fname, monitor='val_loss', verbose=1)
    mtl_model.fit(X_train, [y_train for i in range(n_tasks)],
                  batch_size=100,
                  epochs=FLAGS.epochs,
                  verbose=1,
                  sample_weight=get_mtl_sample_weights(
                      y_train, cohorts_train, all_tasks, sample_weights=samp_weights),
                  callbacks=[early_stopping, checkpointer],
                  validation_data=(X_val, [y_val for i in range(n_tasks)]))

    mtl_model.save(FLAGS.experiment_name + '/models/' +
                   "_".join(model_fname_parts))

    cohort_aucs = []

    y_pred = get_correct_task_mtl_outputs(
        mtl_model.predict(X_val), cohorts_val, all_tasks)

    # task aucs
    for task in all_tasks:
        print('Multitask AUC on', task, ': ')
        if FLAGS.no_val_bootstrap:
            y_pred_in_task = y_pred[cohorts_val == task]
            try:
                auc = roc_auc_score(y_val[cohorts_val == task], y_pred_in_task)
            except:
                auc = np.nan
            cohort_aucs.append(auc)
        else:
            
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, mtl_model,\
                all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            print("(min/max/average):")

        print(cohort_aucs[-1])
        
    # macro average
    cohort_aucs = np.array(cohort_aucs)
    cohort_aucs = np.concatenate(
        (cohort_aucs, np.expand_dims(np.nanmean(cohort_aucs, axis=0), 0)))

    # micro average
    if FLAGS.no_val_bootstrap:
        cohort_aucs = np.concatenate(
        (np.nanmean(cohort_aucs, axis=0), np.array([roc_auc_score(y_val, y_pred)])))
    else:
        min_auc, max_auc, avg_auc = bootstrap_predict(
            X_val, y_val, cohorts_val, 'all', mtl_model,\
            all_tasks=all_tasks, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([[min_auc, max_auc, avg_auc]])))

    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size, FLAGS.num_dense_shared_layers,
                          FLAGS.dense_shared_layer_size, FLAGS.num_multi_layers, FLAGS.multi_layer_size]

    try:
        multitask_model_results = np.load(fname_results)
        multitask_model_key = np.load(fname_keys)
        multitask_model_results = np.concatenate(
            (multitask_model_results, np.expand_dims(cohort_aucs, 0)))
        multitask_model_key = np.concatenate(
            (multitask_model_key, np.array([current_run_params])))

    except:
        multitask_model_results = np.expand_dims(cohort_aucs, 0)
        multitask_model_key = np.array([current_run_params])

    np.save(fname_results, multitask_model_results)
    np.save(fname_keys, multitask_model_key)
    print('Saved multitask results.')

    
def run_global_model(X_train, y_train, cohorts_train,
                     X_val, y_val, cohorts_val,
                     X_test, y_test, cohorts_test,
                     all_tasks, fname_keys, fname_results,
                     FLAGS):
    """
    Train and evaluate global model. 
    Results are saved in FLAGS.experiment_name/results:
        - The numpy file ending in '_keys' contains the parameters for the model, 
          and the numpy file ending in '_results' contains the validation AUCs for that 
          configuration. 
        - If you run multiple configurations for the same experiment name, 
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_global_'. 
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping, 
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual 
          val / test dataset is saved. 
    Args:
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples. 
        cohorts_train (Numpy array): List of cohort membership for each validation example. 
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples. 
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples. 
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        fname_keys (String): filename where the model parameters will be saved.
        fname_results (String): filename where the model AUCs will be saved.
        FLAGS (dictionary): all the arguments.
    """

    model_fname_parts = ['global', 'lstm_shared', str(FLAGS.num_lstm_layers), 'layers', str(FLAGS.lstm_layer_size), 'units',
                         str(FLAGS.num_dense_shared_layers), 'dense_shared', str(FLAGS.dense_shared_layer_size), 
                         'dense_units', 'readmission']

    if FLAGS.test_time:
        model_path = FLAGS.experiment_name + \
            '/models/' + "_".join(model_fname_parts)
        model = load_model(model_path)
        cohort_aucs = []
        y_pred = model.predict(X_test)

        # all bootstrapped AUCs
        for task in all_tasks:
            if FLAGS.test_bootstrap:
                all_aucs = bootstrap_predict(X_test, y_test, cohorts_test, task, model, return_everything=True,
                                             test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
                cohort_aucs.append(np.array(all_aucs))
            else:
                y_pred_in_cohort = y_pred[cohorts_test == task]
                y_true_in_cohort = y_test[cohorts_test == task]
                auc = roc_auc_score(y_true_in_cohort, y_pred_in_cohort)
                cohort_aucs.append(auc)

        if FLAGS.test_bootstrap:
            # Macro AUC
            cohort_aucs = np.array(cohort_aucs)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.expand_dims(np.mean(cohort_aucs, axis=0), 0)))

            # Micro AUC
            all_micro_aucs = bootstrap_predict(X_test, y_test, cohorts_test, 'all', model,
                                               return_everything=True, 
                                               test=True, num_bootstrap_samples=FLAGS.num_test_bootstrap_samples)
            cohort_aucs = np.concatenate(
                (cohort_aucs, np.array([all_micro_aucs])))

        else:
            # Macro AUC
            macro_auc = np.mean(cohort_aucs)
            cohort_aucs.append(macro_auc)

            # Micro AUC
            micro_auc = roc_auc_score(y_test, y_pred)
            cohort_aucs.append(micro_auc)

        suffix = 'single' if not FLAGS.test_bootstrap else 'all'
        test_auc_fname = 'test_auc_on_global_' + suffix
        np.save(FLAGS.experiment_name + '/results/' +
                test_auc_fname, cohort_aucs)
        return

    model = create_single_task_model(FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                                     FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size, X_train.shape[1:], 1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    model_dir = FLAGS.experiment_name + \
        '/checkpoints/' + "_".join(model_fname_parts)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fname = model_dir + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_fname, monitor='val_loss', verbose=1)

    model.fit(X_train, y_train,
              epochs=FLAGS.epochs, batch_size=100,
              sample_weight=samp_weights,
              callbacks=[checkpointer, early_stopping],
              validation_data=(X_val, y_val))

    model.save(FLAGS.experiment_name + '/models/' +
               "_".join(model_fname_parts))

    cohort_aucs = []
    y_pred = model.predict(X_val)
    for task in all_tasks:
        print('Global Model AUC on ', task, ':')
        if FLAGS.no_val_bootstrap:
            try:
                auc = roc_auc_score(
                    y_val[cohorts_val == task], y_pred[cohorts_val == task])
            except:
                auc = np.nan
            cohort_aucs.append(auc)
        else:
            min_auc, max_auc, avg_auc = bootstrap_predict(
                X_val, y_val, cohorts_val, task, model, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
            cohort_aucs.append(np.array([min_auc, max_auc, avg_auc]))
            print ("(min/max/average): ")

        print(cohort_aucs[-1])

    cohort_aucs = np.array(cohort_aucs)

    # Add Macro AUC
    cohort_aucs = np.concatenate(
        (cohort_aucs, np.expand_dims(np.nanmean(cohort_aucs, axis=0), 0)))

    # Add Micro AUC
    if FLAGS.no_val_bootstrap:
        micro_auc = roc_auc_score(y_val, y_pred)
        cohort_aucs = np.concatenate((cohort_aucs, np.array([micro_auc])))
    else:
        min_auc, max_auc, avg_auc = bootstrap_predict(
            X_val, y_val, cohorts_val, 'all', model, num_bootstrap_samples=FLAGS.num_val_bootstrap_samples)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([[min_auc, max_auc, avg_auc]])))

    # Save Results
    current_run_params = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                          FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
    try:
        print('appending results.')
        global_model_results = np.load(fname_results)
        global_model_key = np.load(fname_keys)
        global_model_results = np.concatenate(
            (global_model_results, np.expand_dims(cohort_aucs, 0)))
        global_model_key = np.concatenate(
            (global_model_key, np.array([current_run_params])))

    except Exception as e:
        global_model_results = np.expand_dims(cohort_aucs, 0)
        global_model_key = np.array([current_run_params])

    np.save(fname_results, global_model_results)
    np.save(fname_keys, global_model_key)
    print('Saved global results.')



################ RUN THINGS ####################################################
####################################################################################
if __name__ == "__main__":

    FLAGS = get_args()


    # Make folders for the results & models
    for folder in ['results', 'models', 'checkpoints']:
        if not os.path.exists(os.path.join(FLAGS.experiment_name, folder)):
            os.makedirs(os.path.join(FLAGS.experiment_name, folder))

    # The file that we'll save model configurations to
    sw = 'with_sample_weights' if FLAGS.sample_weights else 'no_sample_weights'
    sw = '' if FLAGS.model_type == 'SEPARATE' else sw
    fname_keys = FLAGS.experiment_name + '/results/' + \
        '_'.join([FLAGS.model_type.lower(), 'model_keys', sw]) + '.npy'
    fname_results = FLAGS.experiment_name + '/results/' + \
        '_'.join([FLAGS.model_type.lower(), 'model_results', sw]) + '.npy'

    # Check that we haven't already run this configuration
    if os.path.exists(fname_keys) and not FLAGS.repeats_allowed:
        model_key = np.load(fname_keys)
        current_run = [FLAGS.num_lstm_layers, FLAGS.lstm_layer_size,
                       FLAGS.num_dense_shared_layers, FLAGS.dense_shared_layer_size]
        if FLAGS.model_type == "MULTITASK":
            current_run = current_run + \
                [FLAGS.num_multi_layers, FLAGS.multi_layer_size]
        print('Now running :', current_run)
        print('Have already run: ', model_key.tolist())
        if current_run in model_key.tolist():
            print('Have already run this configuration. Now skipping this one.')
            sys.exit(0)

    # Load Data
    try:
        X = np.load(os.path.join(FLAGS.data_path ,'X.npy'))
        Y = np.load(os.path.join(FLAGS.data_path ,'Y.npy'))
        Y = Y.astype(int)
        
        careunits = np.load(os.path.join(FLAGS.data_path ,'careunits.npy'), allow_pickle=True)
        subject_ids = np.load(os.path.join(FLAGS.data_path ,'subject_ids.npy'), allow_pickle=True)
        print('LOADING:\n\tModel ready datasets already exists')
    except Exception as e:
        conn = postgre_connect(FLAGS.db_config)
        
        print('START:\n\tSucessfully created connection to Postgres')
    
        # Queries
        timeseries = "select * from mimiciii.timseries_table_avg;"
        demographics = "select * from mimiciii.readmission;"
        comorbidities = "select * from mimiciii.comorbidities_table;"
        INDEX_COLS = ['subject_id', 'icustay_id', 'hadm_id', 'hours_in']

        X, Y, careunits, subject_ids = load_processed_data(FLAGS.data_path, 
                                                           demographics, comorbidities, 
                                                           conn, INDEX_COLS)
        Y = Y.astype(int)
      

    # Split
    if FLAGS.cohorts == 'careunit':
        cohort_col = careunits
        print(f'COMPLETE:\n\t{FLAGS.cohorts} added to the dataset as a feature')
    elif FLAGS.cohorts == 'custom':
        cohort_col = np.load('cluster_membership/' + FLAGS.cohort_filepath)
        cohort_col = np.array([str(c) for c in cohort_col])
        print(f'COMPLETE:\n\t{FLAGS.cohorts} added to the dataset as a feature')

    # Include cohort membership as an additional feature
    if FLAGS.include_cohort_as_feature:
        cohort_col_onehot = pd.get_dummies(cohort_col).as_matrix()
        cohort_col_onehot = np.expand_dims(cohort_col_onehot, axis=1)
        cohort_col_onehot = np.tile(cohort_col_onehot, (1, 24, 1))
        X = np.concatenate((X, cohort_col_onehot), axis=-1)
        print(f'COMPLETE:\n\t{FLAGS.cohorts} added to the dataset and one-hot encoded')

    # Train, val, test split
    X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        cohorts_train, cohorts_val, cohorts_test = stratified_split(
            X, Y, cohort_col, train_val_random_seed=FLAGS.train_val_random_seed)
    print(f'COMPLETE:\n\tStratified train/test/val split')

    # Sample Weights
    task_weights = dict()
    all_tasks = np.unique(cohorts_train)
    for cohort in all_tasks:
        num_in_cohort = len(np.where(cohorts_train == cohort)[0])
        print("Number of people in each cluster in the training set " +
              str(cohort) + ": " + str(num_in_cohort))
        task_weights[cohort] = len(X_train)*1.0/num_in_cohort

    if FLAGS.sample_weights:
        samp_weights = np.array([task_weights[cohort]
                                 for cohort in cohorts_train])

    else:
        samp_weights = None

    # Run model
    run_model_args = [X_train, y_train, cohorts_train,
                      X_val, y_val, cohorts_val,
                      X_test, y_test, cohorts_test,
                      all_tasks, fname_keys, fname_results,
                      FLAGS]

    if FLAGS.model_type == 'SEPARATE':
        run_separate_models(*run_model_args)
    elif FLAGS.model_type == 'GLOBAL':
        run_global_model(*run_model_args)
    if FLAGS.model_type == 'MULTITASK':
        run_multitask_model(*run_model_args)