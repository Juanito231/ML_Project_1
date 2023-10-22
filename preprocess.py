import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import pickle

# Utility functions

def train_validation_split(data, ratio, seed):
    """Split data into training and validation set."""
    np.random.seed(seed)
    np.random.shuffle(data)
    split_index = int(len(data) * ratio)
    return data[:split_index], data[split_index:]

def k_fold_split(data, k, seed):
    """Split data into k folds."""
    np.random.seed(seed)
    np.random.shuffle(data)
    return np.array_split(data, k)

def standardize_data(data):
    """Standardize data."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / std

def removing_features(number_NaN, list_ids, data, threshold=0.1):
    to_remove=[]
    for i in range(len(number_NaN)):
        if number_NaN[i] > round(len(data))*threshold:
            to_remove.append(i)
    reduced_data = np.delete(data, to_remove, axis = 1)
    reduced_ids = np.delete(list_ids, to_remove, axis = 0)
    return reduced_data, reduced_ids

def remove_identic_col(list_ids, x):
    """remove the column that have the same value everywhere to avoid std = 0 after and because not useful"""
    to_remove=[]
    for i, col in enumerate(x.T):
        if len(np.unique(col))==1:
            to_remove.append(i)
    reduced_data = np.delete(x, to_remove, axis = 1)
    reduced_ids = np.delete(list_ids, to_remove, axis = 0)
    return reduced_data, reduced_ids

def clean_outliers(X):
    for column in range(X.shape[1]):  # for each feature:
        # calculating 25th and 75th quantiles
        q1_X = np.nanquantile(X[:, column], 0.25, axis=0)
        q3_X = np.nanquantile(X[:, column], 0.75, axis=0)
        IQR_X = q3_X - q1_X  # inter quantile range
        # calculating lower and upper bounds
        lower_bound = q1_X - 1.5 * IQR_X
        upper_bound = q3_X + 1.5 * IQR_X
        # finding which observations are outside/inside of the bounds
        above = X[:, column] > upper_bound
        below = X[:, column] < lower_bound
        outside = above | below
        inside = np.invert(outside)
        # calculate median value of observations that are inside boundaries
        median = np.median(X[inside, column])
        # setting outliers equal to median
        X[outside, column] = median
    return X


def replace_NaN_mean_column(column):
    # we already standardized the data so the mean is 0
    for i in range(len(column)):
        if np.isnan(column[i]):
            column[i]= 0
    return column

def replace_NaN_median_column(column):
    median=np.nanmedian(column)
    for i in range(len(column)):
        if np.isnan(column[i]):
            column[i]=median
    return column

def replace_NaN(matrix, method='mean'): 
    for i in range(matrix.shape[1]):
        if method=='mean':
            matrix[:,i] = replace_NaN_mean_column(matrix[:,i])
        elif method=='median':
            matrix[:,i] = replace_NaN_median_column(matrix[:,i])
    return matrix

