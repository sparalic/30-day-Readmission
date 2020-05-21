#!usr/bin/env python3

'''
30 day Readmission after ICU stay using MIMIC III
modified from original paper repo: https://github.com/mit-ddig/multitask-patients

@author: Sparkle Russell-Puleri
@date: May, 9th 2020

'''
from __future__ import absolute_import
from __future__ import print_function

from numpy.random import seed
seed(1)
import os
import sys
import argparse
import numpy as np
import pandas as pd
import psycopg2
import pandas.io.sql as sqlio
from datetime import datetime as dt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json



def postgre_connect(db_login):
    """ 
    Loads database json credentials files 
    Args:
      db_login(str): file path to json file 
    Returns: Database connection
        conn: 
    """
    db_conn = json.loads(open(db_login, 'r').read())
    conn = psycopg2.connect(host=db_conn['host'],
                            port = db_conn['port'], 
                            database=db_conn["database"], 
                            user=db_conn['user'], 
                            password=db_conn['password'])
    return conn


def process_timeseries_data(time_series, data_path, conn, INDEX_COLS):
     """ 
    Loads timeseries data set and preprocesses, saves and returns it
    Args:
      time_series: SQL query for time series data
      data_path(str): SQL query for the readmission/demograhics table
      conn(object): Database connection 
      INDEX_COLS(list): list of index columns
    Returns:
        X: Pandas DataFrame containing one row per patient per hour. 
           Each row should include the columns {'subject_id', 'icustay_id', 'hours_in', 'hadm_id'}
           along with any additional features.
       
    """
        
     pivot_col = 'label'
     values = 'valuenum_avg'
    
     df = sqlio.read_sql_query(time_series, conn)
    
     df = df.pivot_table(index=INDEX_COLS, 
                              columns=pivot_col, 
                              values=values).reset_index()
     print('COMPLETE:\n\tSucessfully pivoted labels column\n')
    
     df.to_hdf(os.path.join(data_path, 'X.h5'), key='X')
     return df

def load_data(data_path, demo, comorb, conn):
    """ 
    Loads X, Y, and static matrices into Pandas DataFrames 
    Args:
      query: SQL query for demographics data
      demo(str): SQL query for the readmission/demograhics table
      comorb(str): SQL query for the comorbidities table
      conn(object): Database connection 
    Returns:
        X: Pandas DataFrame containing one row per patient per hour. 
           Each row should include the columns {'subject_id', 'icustay_id', 'hours_in', 'hadm_id'}
           along with any additional features.
        demographics: Pandas DataFrame containing one row per patient. 
                Should include {'subject_id', 'hadm_id', 'icustay_id'}.
        comborbidites: Pandas DataFrame containing one row per patient. 
                Should include {'subject_id', 'hadm_id', 'icustay_id'}.
    """

    X = pd.read_hdf(os.path.join(data_path,'X.h5'))
    demographics = sqlio.read_sql_query(demo, conn)
    comborbidites = sqlio.read_sql_query(comorb, conn)
  
    return X, demographics, comborbidites

def make_discrete_values(mat, INDEX_COLS):
    """ 
    Converts numerical values into one-hot vectors of number of z-scores 
    above/below the mean, aka physiological words (see Suresh et al 2017).
    Args:
        mat (Pandas DataFrame): Matrix of feature values including columns in
        INDEX_COLS(list): list of index columns
    Returns:
        DataFrame: X_categorized. A DataFrame where each features is a set of
        indicator columns signifying number of z-scores above or below the mean.
    """

    normal_dict = mat.groupby(['subject_id']).mean().mean().to_dict()
    std_dict = mat.std().to_dict()
    feature_cols = mat.columns[len(INDEX_COLS):]
    print(feature_cols)
    X_words = mat.loc[:, feature_cols].apply(
        lambda x: transform_vals(x, normal_dict, std_dict), axis=0)
    mat.loc[:, feature_cols] = X_words
    X_categorized = pd.get_dummies(mat, columns=mat.columns[len(INDEX_COLS):])
    na_columns = [col for col in X_categorized.columns if '_9' in col]
    X_categorized.drop(na_columns, axis=1, inplace=True)
    return X_categorized


def transform_vals(x, normal_dict, std_dict):
    """ 
    Helper function to convert values to z-scores between -4 and 4. 
    Missing values are assigned 9. 
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    x = 1.0*(x - normal_dict[x.name])/std_dict[x.name]
    x = x.round()
    x = x.clip(-4, 4)
    x = x.fillna(9)
    x = x.round(0).astype(int)
    return x
    
def categorize_age(age):
    """ 
    Categorize age into windows. 
    Args:
        age (int): A number.
    Returns:
        int: cat. The age category.
    """

    if age > 10 and age <= 30:
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else:
        cat = 4
    return cat

def categorize_los(los):
    """ 
    Categorize Length of stay(LOS) into 4 windows. 
    Args:
        los (int): A number.
    Returns:
        int: cat. The los category.
    """

    if los > 1 and los <= 15:
        cat = 1
    elif los > 16 and los <= 30:
        cat = 2
    elif los > 31 and los <= 45:
        cat = 3
    else:
        cat = 4
    return cat

def categorize_ethnicity(ethnicity):
    """ 
    Groups ethnicity sub-categories into 5 major categories.
    Args:
        ethnicity (str): string indicating patient ethnicity.
    Returns:
        string: ethnicity. Categorized into 5 main categories. 
    """

    if 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else:
        ethnicity = 'OTHER'
    return ethnicity


def _pad_df(df, data_hours, pad_value=np.nan):
    """ Add dataframe with padding for the last 48 hours of each patients stay"""

    existing = set(df.index.get_level_values(1))
    fill_hrs = set(range(int(max(existing))-data_hours, int(max(existing)))) - existing
    if len(fill_hrs) > 0:
        return fill_hrs
    else:
        return 0


def load_processed_data(data_path, demo, comorb,  conn, INDEX_COLS, data_hours=48):
    """
    Either read pre-processed data from a saved folder, or load in the raw data and preprocess it.
    Should have the files 'saps.csv' (with columns 'subject_id', 'hadm_id', 'icustay_id')
    and 'code_status.csv' (with columns 'subject_id', 'hadm_id', 'icustay_id')
    in the local directory.
    
    Args: 
        data_path(str): file path to physiological data
        demo(str): SQL query for the readmission/demograhics table
        comorb(str): SQL query for the comorbidities table
        conn(object): Database connection 
        INDEX_COLS(list): list of index columns
        data_hours (int): hours of data to use for predictions.
        
    Returns: 
        X (Numpy array): matrix of data of size n_samples x n_timesteps x n_features.  
        Y (Numpy array): binary array of len n_samples corresponding to in hospital mortality after the gap time.
        careunits (Numpy array): array w careunit membership of each sample.
        subject_ids (Numpy array): subject_ids corresponding to each row of the X/Y/careunits/saps_quartile arrays.
    """
    save_data_path = 'data/readmissions_' + str(data_hours) + '/'
 

    # see if we already have the data matrices saved
    try:
        X = np.load(data_path + 'X.npy')
        careunits = np.load(save_data_path + 'careunits.npy')
        subject_ids = np.load(save_data_path + 'subject_ids.npy')
        Y = np.load(save_data_path + 'Y2.npy')
        print('Loaded data from ' + save_data_path)
        print('shape of X: ', X.shape)

    # otherwise create them
    except Exception as e:
        
        X, static, comorb = load_data(data_path, demo, comorb, conn)
        
        print('COMPLETE:\n\tSucessfully loaded static data\n')
        
        # Find the last hour of each patients stay
        max_time = X.groupby('subject_id').\
                   max().reset_index().\
                   rename(columns={'subject_id':'subject_id','hours_in':'max_time'})[['subject_id', 'max_time']]
        
        X = X.merge(max_time,how='left', on='subject_id')
        # remove pats who stayed less than 48 hours
        X = X[X['max_time']>=data_hours]

        #Get last 48 hours of stay by patient
        X = X[X['hours_in']>= X['max_time']-data_hours]
        
        #Remove max time column
        X.drop('max_time', axis=1, inplace=True)
        
        print('COMPLETE:\n\tSucessfully filtered last 48 hrs of each patients stay')
        
        # Make discrete values and cut off last 48 hrs of stay
        X_discrete = make_discrete_values(X, INDEX_COLS)
        X_discrete = X_discrete.groupby(['subject_id', 'hours_in']).max().reset_index()
        X_discrete = X_discrete[[
            c for c in X_discrete.columns if c not in ['hadm_id', 'icustay_id']]]
        
        print('COMPLETE:\n\tSucessfully discretized the features')
        
        # Pad people whose records stop early
        test = X_discrete.set_index(['subject_id', 'hours_in'])
        extra_hours = test.groupby(level=0).apply(_pad_df, data_hours)
        extra_hours = extra_hours[extra_hours != 0].reset_index()
        extra_hours.columns = ['subject_id', 'pad_hrs']
        pad_tuples = []
        for s in extra_hours.subject_id:
            for hr in list(extra_hours[extra_hours.subject_id == s].pad_hrs)[0]:
                pad_tuples.append((s, hr))
        pad_df = pd.DataFrame(np.nan, index=pd.MultiIndex.from_tuples(
            pad_tuples, names=('subject_id', 'hours_in')), columns=test.columns)
        new_df = pd.concat([test, pad_df], axis=0)
        
        print('COMPLETE:\n\tSucessfully padded to last 48 hrs')

        # Get the last 48 consecutive hours, backfill and forward fill from the first non missing row
        new_df = new_df.sort_index().bfill().reset_index()
        new_df = new_df.groupby('subject_id').apply(lambda x: x.iloc[-data_hours:, :]).reset_index(drop=True)
        
        print('COMPLETE:\n\tSucessfully backfilled missing data to last 48 hrs')
        
        # get the static vars we want, make them discrete columns
        # Remove patients who died
        static = static[static['hospital_expire_flag']!=1]
        static_to_keep = static[['subject_id', 'gender', 'age', 'ethnicity',
                                 'insurance', 'discharge_location','language',
                                 'los', 'first_careunit', 'readmission_30_days']]
        static_to_keep['language'].replace(to_replace=[None], value='Missing', inplace=True)
        static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(
            categorize_ethnicity)
        static_to_keep.loc[:, 'age'] = static_to_keep['age'].apply(
            categorize_age)
        static_to_keep.loc[:, 'los'] = static_to_keep['los'].apply(
            categorize_los)
        static_to_keep.loc[:, 'language'] = static_to_keep['language'].apply(
            categorize_language)
        static_to_keep = pd.get_dummies(static_to_keep, columns=[
                                        'gender', 'age', 
                                        'ethnicity', 'los',
                                        'language','insurance', 'discharge_location'])
        print('COMPLETE:\n\tSucessfully dummy continuous and categorical features')
        
         # merge the phys with static
        X_full = pd.merge(new_df, static_to_keep,
                          on='subject_id', how='inner')
        comorb_cols = comorb[[col for col in comorb.columns if col not in ['hadm_id']]]
        X_full = pd.merge(X_full, comorb_cols, on='subject_id', how='inner')
        X_full = X_full.set_index(['subject_id', 'hours_in'])
        
        print('COMPLETE:\n\tSucessfully merged time series data with comorbidities and demographics data')
        
        # print readmission per careunit
        readmission_by_careunit = X_full.groupby(
            'subject_id')['first_careunit', 'readmission_30_days'].max()
        for cu in readmission_by_careunit.first_careunit.unique():
            print(cu + ": " + str(np.sum(readmission_by_careunit[readmission_by_careunit.first_careunit == cu].\
                  readmission_30_days)) + ' out of ' + str(len(readmission_by_careunit[readmission_by_careunit.\
                  first_careunit == cu])))
            
            
        # create Y and cohort matrices
        subject_ids = X_full.index.get_level_values(0).unique()
        Y = X_full[['readmission_30_days']].groupby(level=0).max()
        careunits = X_full[['first_careunit']].groupby(level=0).max()
        Y = Y.reindex(subject_ids)
        careunits = careunits.reindex(subject_ids)
        
        
        # delete those columns from the X matrix
        X_full = X_full.loc[:, X_full.columns != 'readmission_30_days']
        X_full = X_full.loc[:, X_full.columns != 'first_careunit']
        
        
        feature_names = X_full.columns
        

        # get the data as a np matrix of size num_examples x timesteps x features
        X_full_matrix = np.reshape(
            X_full.to_numpy(), (len(subject_ids), data_hours, -1))
        print("COMPLETE: matrix of size num_examples x timesteps x features, shape of X: ")
        print(X_full_matrix.shape)

        # print feature values
        print("Features : ")
        print(np.array(X_full.columns))

        print(subject_ids)
        print(Y.index)
        print(careunits.index)

        print("Number of positive examples : ", len(Y[Y == 1]))

        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        
        np.save(save_data_path + 'feature_names.npy', feature_names)

        np.save(save_data_path + 'X.npy', X_full_matrix)
        np.save(save_data_path + 'careunits.npy',
                np.squeeze(careunits.to_numpy(), axis=1))
        np.save(save_data_path + 'subject_ids.npy', np.array(subject_ids))
        np.save(save_data_path + 'Y.npy', np.squeeze(Y.to_numpy(), axis=1))

        X = X_full_matrix

    return X, Y, careunits, subject_ids



def stratified_split(X, Y, cohorts, train_val_random_seed=0):
    """ 
    Return stratified split of X, Y, and a cohort membership array, stratified by outcome. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        Y (Numpy array): Y matrix, shape = num_patients.
        cohorts (Numpy array): array of cohort membership, shape = num_patients.
        train_val_random_seed (int): random seed for splitting.
    Returns:
        Numpy arrays: X_train, X_val, X_test, y_train, y_val, y_test, 
        cohorts_train, cohorts_val, cohorts_test. 
    """

    X_train_val, X_test, y_train_val, y_test, \
        cohorts_train_val, cohorts_test = \
        train_test_split(X, Y, cohorts, test_size=0.2,
                         random_state=train_val_random_seed, stratify=Y)

    X_train, X_val, y_train, y_val, \
        cohorts_train, cohorts_val = \
        train_test_split(X_train_val, y_train_val, cohorts_train_val, test_size=0.125,
                         random_state=train_val_random_seed, stratify=y_train_val)

    return X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        cohorts_train, cohorts_val, cohorts_test

def generate_bootstrap_indices(X, y, split, num_bootstrap_samples=100):
    """ 
    Generates and saves to file sets of indices for val or test bootstrapping. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        split (string): 'val' or 'test' indicating for which split to generate indices.
        num_bootstrap_samples (int): number indicating how many sets of bootstrap samples to generate.
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    all_pos_samples = []
    all_neg_samples = []
    for i in range(num_bootstrap_samples):
        pos_samples = np.random.choice(
            len(positive_X), replace=True, size=len(positive_X))
        neg_samples = np.random.choice(
            len(negative_X), replace=True, size=len(negative_X))
        all_pos_samples.append(pos_samples)
        all_neg_samples.append(neg_samples)

    np.save(split + '_pos_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_pos_samples))
    np.save(split + '_neg_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_neg_samples))
    return all_pos_samples, all_neg_samples


def get_bootstrapped_dataset(X, y, cohorts, index=0, test=False, num_bootstrap_samples=100):
    """ 
    Returns a bootstrapped (sampled w replacement) dataset. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        cohorts (Numpy array): array of cohort membership, shape = num_patients.
        index (int): which bootstrap sample to look at. 
        test (bool): 
        num_bootstrap_samples (int):
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    if index == 0:
        return X, y, cohorts

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    positive_cohorts = cohorts[np.where(y == 1)]
    negative_cohorts = cohorts[np.where(y == 0)]
    positive_y = y[np.where(y == 1)]
    negative_y = y[np.where(y == 0)]

    split = 'test' if test else 'val'
    try:
        pos_samples = np.load(
            split + '_pos_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
        neg_samples = np.load(
            split + '_neg_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
    except:
        all_pos_samples, all_neg_samples = generate_bootstrap_indices(
            X, y, split, num_bootstrap_samples)
        pos_samples = all_pos_samples[index]
        neg_samples = all_neg_samples[index]

    positive_X_bootstrapped = positive_X[pos_samples]
    negative_X_bootstrapped = negative_X[neg_samples]
    all_X_bootstrappped = np.concatenate(
        (positive_X_bootstrapped, negative_X_bootstrapped))
    all_y_bootstrapped = np.concatenate(
        (positive_y[pos_samples], negative_y[neg_samples]))
    all_cohorts_bootstrapped = np.concatenate(
        (positive_cohorts[pos_samples], negative_cohorts[neg_samples]))

    return all_X_bootstrappped, all_y_bootstrapped, all_cohorts_bootstrapped

def flatten(X):
    '''Flattens a 3D array 
    Args: 
      X: 3D array for lstm, where the array is sample x timesteps x features.
    Returns: 
      flattened_X  A 2D array, sample x features.
    source: https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)


def generate_bootstrap_indices(X, y, split, num_bootstrap_samples=100):
    """ 
    Generates and saves to file sets of indices for val or test bootstrapping. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        split (string): 'val' or 'test' indicating for which split to generate indices.
        num_bootstrap_samples (int): number indicating how many sets of bootstrap samples to generate.
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    all_pos_samples = []
    all_neg_samples = []
    for i in range(num_bootstrap_samples):
        pos_samples = np.random.choice(
            len(positive_X), replace=True, size=len(positive_X))
        neg_samples = np.random.choice(
            len(negative_X), replace=True, size=len(negative_X))
        all_pos_samples.append(pos_samples)
        all_neg_samples.append(neg_samples)

    np.save(split + '_pos_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_pos_samples))
    np.save(split + '_neg_bootstrap_samples_' +
            str(num_bootstrap_samples), np.array(all_neg_samples))
    return all_pos_samples, all_neg_samples


def get_bootstrapped_dataset(X, y, cohorts, index=0, test=False, num_bootstrap_samples=100):
    """ 
    Returns a bootstrapped (sampled w replacement) dataset. 
    Args:
        X (Numpy array): X matrix, shape = num patients x num timesteps x num features.
        y (Numpy array): Y matrix, shape = num_patients.
        cohorts (Numpy array): array of cohort membership, shape = num_patients.
        index (int): which bootstrap sample to look at. 
        test (bool): 
        num_bootstrap_samples (int):
    Returns:
        Numpy arrays: all_pos_samples, all_neg_samples. Contains num_bootstrap_samples indices 
        of positive and negative examples. 
    """

    if index == 0:
        return X, y, cohorts

    positive_X = X[np.where(y == 1)]
    negative_X = X[np.where(y == 0)]
    positive_cohorts = cohorts[np.where(y == 1)]
    negative_cohorts = cohorts[np.where(y == 0)]
    positive_y = y[np.where(y == 1)]
    negative_y = y[np.where(y == 0)]

    split = 'test' if test else 'val'
    try:
        pos_samples = np.load(
            split + '_pos_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
        neg_samples = np.load(
            split + '_neg_bootstrap_samples_' + str(num_bootstrap_samples) + '.npy')[index]
    except:
        all_pos_samples, all_neg_samples = generate_bootstrap_indices(
            X, y, split, num_bootstrap_samples)
        pos_samples = all_pos_samples[index]
        neg_samples = all_neg_samples[index]

    positive_X_bootstrapped = positive_X[pos_samples]
    negative_X_bootstrapped = negative_X[neg_samples]
    all_X_bootstrappped = np.concatenate(
        (positive_X_bootstrapped, negative_X_bootstrapped))
    all_y_bootstrapped = np.concatenate(
        (positive_y[pos_samples], negative_y[neg_samples]))
    all_cohorts_bootstrapped = np.concatenate(
        (positive_cohorts[pos_samples], negative_cohorts[neg_samples]))

    return all_X_bootstrappped, all_y_bootstrapped, all_cohorts_bootstrapped

def bootstrap_predict(X_orig, y_orig, cohorts_orig, task, model, return_everything=False, test=False, all_tasks=[], num_bootstrap_samples=100):
    """ 
    Evaluates model on each of the num_bootstrap_samples sets. 
    Args: 
        X_orig (Numpy array): The X matrix.
        y_orig (Numpy array): The y matrix. 
        cohorts_orig (Numpy array): List of cohort membership for each X example.
        task (String/Int): task to evalute on (either 'all' to evalute on the entire dataset, or a specific task). 
        model (Keras model): the model to evaluate.
        return_everything (bool): if True, return list of AUCs on all bootstrapped samples. If False, return [min auc, max auc, avg auc].
        test (bool): if True, use the test bootstrap indices.
        all_tasks (list): list of the tasks (used for evaluating multitask model).
        num_bootstrap_samples (int): number of bootstrapped samples to evalute on.
    Returns: 
        all_aucs OR min_auc, max_auc, avg_auc depending on the value of return_everything.
    """

    all_aucs = []

    for i in range(num_bootstrap_samples):
        X_bootstrap_sample, y_bootstrap_sample, cohorts_bootstrap_sample = get_bootstrapped_dataset(
            X_orig, y_orig, cohorts_orig, index=i, test=test, num_bootstrap_samples=num_bootstrap_samples)
        if task != 'all':
            X_bootstrap_sample_task = X_bootstrap_sample[cohorts_bootstrap_sample == task]
            y_bootstrap_sample_task = y_bootstrap_sample[cohorts_bootstrap_sample == task]
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample[cohorts_bootstrap_sample == task]
        else:
            X_bootstrap_sample_task = X_bootstrap_sample
            y_bootstrap_sample_task = y_bootstrap_sample
            cohorts_bootstrap_sample_task = cohorts_bootstrap_sample

        preds = model.predict(X_bootstrap_sample_task, batch_size=128)
        if len(preds) < len(y_bootstrap_sample_task):
            preds = get_correct_task_mtl_outputs(
                preds, cohorts_bootstrap_sample_task, all_tasks)

        try:
            auc = roc_auc_score(y_bootstrap_sample_task, preds)
            all_aucs.append(auc)
           
     
        except Exception as e:
            print(e)
            print('Skipped this sample.')
        
    avg_auc = np.mean(all_aucs)
    min_auc = min(all_aucs)
    max_auc = max(all_aucs)

    if return_everything:
        return all_aucs
    else:
        return min_auc, max_auc, avg_auc
    
def get_mtl_sample_weights(y, cohorts, all_tasks, sample_weights=True):
    """ 
    Generates a dictionary of sample weights for the multitask model that masks out 
    (and prevents training on) outputs corresponding to cohorts to which a given sample doesn't belong. 
    Args: 
        y (Numpy array): The y matrix.
        cohorts (Numpy array): cohort membership corresponding to each example, in the same order as y.
        all_tasks (list/Numpy array): list of all unique tasks.
        sample_weights (list/Numpy array): if samples should be weighted differently during training, 
                                           provide a list w len == num_samples where each value is how much 
                                           that value should be weighted.
    Returns: 
        sw_dict (dictionary): Dictionary mapping task to list w len == num_samples, where each value is 0 if 
                              the corresponding example does not belong to that task, and either 1 or a sample weight
                              value (if sample_weights != None) if it does.
    """

    sw_dict = {}
    for task in all_tasks:
        task_indicator_col = (cohorts == task).astype(int)
        if sample_weights:
            task_indicator_col = np.array(
                task_indicator_col) * np.array(sample_weights)
        sw_dict[task] = task_indicator_col
    return sw_dict


def get_correct_task_mtl_outputs(mtl_output, cohorts, all_tasks):
    """ 
    Gets the output corresponding to the right task given the multitask output.  Necessary since 
    the MTL model will produce an output for each cohort's output, but we only care about the one the example
    actually belongs to. 
    Args: 
        mtl_output (Numpy array/list): the output of the multitask model. Should be of size n_tasks x n_samples.
        cohorts (Numpy array): list of cohort membership for each sample.
        all_tasks (list): unique list of tasks (should be in the same order that corresponds with that of the MTL model output.)
    Returns:
        mtl_output (Numpy array): an array of size n_samples x 1 where each value corresponds to the MTL model's
                                  prediction for the task that that sample belongs to.
    """

    n_tasks = len(all_tasks)
    cohort_key = dict(zip(all_tasks, range(n_tasks)))
    mtl_output = np.array(mtl_output)
    mtl_output = mtl_output[[cohort_key[c]
                             for c in cohorts], np.arange(len(cohorts))]
    return mtl_output

    
    