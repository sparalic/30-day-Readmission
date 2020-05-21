
#!usr/bin/env python3

'''
30 day Readmission after ICU stay using MIMIC III
modified from original paper repo: https://github.com/mit-ddig/multitask-patients

@author: Sparkle Russell-Puleri
@date: May, 9th 2020

'''
from __future__ import absolute_import
from __future__ import print_function

from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.mixture import GaussianMixture
import pickle
from numpy.random import seed
from preprocessing import *
seed(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=50, \
        help='The embedding size, or latent dimension of the autoencoder. Type: int. Default: 50.')
    parser.add_argument("--ae_epochs", type=int, default=100, \
        help='Number of epochs to train autoencoder. Type: int. Default: 100.')
    parser.add_argument("--ae_learning_rate", type=float, default=0.0001, \
        help='Learning rate for autoencoder. Type: float. Default: 0.0001.')
    parser.add_argument("--num_clusters", type=int, default=4, \
        help='Number of clusters for GMM. Type: int. Default: 3.')
    parser.add_argument("--n_init", type=int, default=30, \
        help='Number of intializations for GMM. Type: int. Default: 30.')
    parser.add_argument("--gmm_tol", type=float, default=0.0001,
        help='The convergence threshold for the GMM. Type: float. Default: 0.0001.')
    parser.add_argument("--data_hours", type=int, default=24, \
        help='The number of hours of data to use. \
        Type: int. Default: 24.')
    parser.add_argument("--gap_time", type=int, default=12, help="Gap between data and when predictions are made. Type: int. Default: 12.")
    parser.add_argument("--save_to_fname", type=str, default='test_clusters.npy', \
        help="Filename to save cluster memberships to. Type: String. Default: 'test_clusters.npy'")
    parser.add_argument("--train_val_random_seed", type=int, default=0, \
        help="Random seed to use during train / val / split process. Type: int. Default: 0.")
    parser.add_argument('--db_config', '-db', type=str, help='File path to the databse JSON file',
                        default='../db_login.JSON')
    parser.add_argument('--data_path',  type=str, help='Path to save physiological data',  default='data/')
    parser.add_argument('--save_data_path',  type=str, help='Path to save outputs',  default='data/')
    parser.add_argument('--conn',  type=str, help='Database connection',  default='conn')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print(args)
    return args


########## CREATE AE MODEL ###############################################################
##########################################################################################

def create_seq_ae(X_train, X_val, latent_dim, learning_rate):
    """
    Build sequence autoencoder. 
    Args: 
        X_train (Numpy array): training data. (shape = n_samples x n_timesteps x n_features)
        X_val (Numpy array): validation data.
        latent_dim (int): hidden representation dimension.
        learning_rate (float): learning rate for training.
    Returns: 
        encoder (Keras model): compiled model that takes original data as input and produces representation.
        sequence_autoencoder (Keras model): compiled autoencoder model.
    """

    timesteps = X_train.shape[1]
    input_dim = X_train.shape[2]
    latent_dim = latent_dim

    inputs = tf.keras.Input(shape=(timesteps, input_dim))
    encoded = tf.keras.layers.LSTM(latent_dim)(inputs)

    decoded = tf.keras.layers.RepeatVector(timesteps)(encoded)
    decoded = tf.keras.layers.LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)


    sequence_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                                 loss='mse')

    return encoder, sequence_autoencoder


########## RUN AE MODEL ##################################################################
##########################################################################################

def train_seq_ae(X_train, X_val, FLAGS):
    """
    Train a sequence to sequence autoencoder.
    Args: 
        X_train (Numpy array): training data. (shape = n_samples x n_timesteps x n_features)
        X_val (Numpy array): validation data.
        FLAGS (dictionary): all provided arguments.
    Returns: 
        encoder (Keras model): trained model to encode to latent space.
        sequence autoencoer (Keras model): trained autoencoder.
    """
    encoder, sequence_autoencoder = create_seq_ae(X_train, X_val, FLAGS.latent_dim, FLAGS.ae_learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # fit the model
    print("Fitting Sequence Autoencoder ... ")
    sequence_autoencoder.fit(X_train, X_train,
                    epochs=FLAGS.ae_epochs,
                    batch_size=128,
                    shuffle=True,
                    callbacks=[early_stopping],
                    validation_data=(X_val, X_val))


    if not os.path.exists('clustering_models/'):
        os.makedirs('clustering_models/')

    encoder.save('clustering_models/encoder_' + str(FLAGS.data_hours))
    sequence_autoencoder.save('clustering_models/seq_ae_' + str(FLAGS.data_hours))
    return encoder, sequence_autoencoder

def fix_byte_data(s):
    """ Python 2/3 fix """

    try:
        s = s.decode()
    except AttributeError:
        pass
    return s

def main():  
    
    FLAGS = get_args()
    
    try:
        os.makedirs(FLAGS.data_path)
    except:
        pass
    

    conn = postgre_connect(FLAGS.db_config)
    
    if FLAGS.verbose:
        print('START:\n\tSucessfully created connection to Postgres')

    # Queries
    timeseries = "select * from mimiciii.timseries_table_avg;"
    demographics = "select * from mimiciii.readmission;"
    comorbidities = "select * from mimiciii.comorbidities_table;"
    INDEX_COLS = ['subject_id', 'icustay_id', 'hadm_id', 'hours_in']
        
    try:
        timeseries = pd.read_hdf(os.path.join(FLAGS.data_path,'X.h5'))
        print('LOADING:\n\tX5 file for phsiological and vitals data already exists')
        
    except Exception as e:   
        timeseries = process_timeseries_data(timeseries, FLAGS.data_path, conn, INDEX_COLS)
        print('COMPLETE:\n\tSucessfully processed phsiological and vitals data')
     
    
    
    X, Y, careunits, subject_ids = load_processed_data(FLAGS.data_path, demographics, comorbidities, conn, INDEX_COLS)
    print('COMPLETE:\n\tSucessfully processed data for modeling')
    
    Y = Y.astype(int)
    cohort_col = careunits

    # Train, val, test split
    X_train, X_val, X_test, \
    y_train, y_val, y_test,cohorts_train, cohorts_val,\
    cohorts_test = stratified_split(X, Y, cohort_col, train_val_random_seed=FLAGS.train_val_random_seed)

    # Train autoencoder
    encoder, sequence_autoencoder = train_seq_ae(X_train, X_val, FLAGS)

    # Get Embeddings
    embedded_train = encoder.predict(X_train)
    embedded_all = encoder.predict(X)
    print("COMPLETE: Feature embedding with autoencoder ...")
    
    # Train GMM
    print("Fitting GMM ...")
    gm = GaussianMixture(n_components=FLAGS.num_clusters, tol=FLAGS.gmm_tol, n_init=FLAGS.n_init, verbose=True)
    gm.fit(embedded_train)
    pickle.dump(gm, open('clustering_models/gmm_' + str(FLAGS.data_hours), 'wb'))

    # Get cluster membership
    cluster_preds = gm.predict(embedded_all)

    if not os.path.exists('cluster_membership/'):
        os.makedirs('cluster_membership/')
    np.save('cluster_membership/' + FLAGS.save_to_fname, cluster_preds)
    print(f"COMPLETE: Cluster membership created and saved{FLAGS.save_to_fname}")

if __name__ == '__main__':
    main()


    