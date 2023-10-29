import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import pickle

# Utility functions

def load_train_data():
    """load data."""
    f = open(f"dataset/x_train.csv")
    features = f.readline()
    feature_names = features.split(',')
    data = np.loadtxt(f"dataset/x_train.csv", delimiter=",", skiprows=1, dtype=str)
    return data,feature_names

def convert_row_to_float(row):
    """Convert values in row to float or np.nan."""
    new_row = []
    for item in row:
        try:
            new_row.append(float(item))
        except ValueError:
            new_row.append(np.nan)
    return np.array(new_row)

def convert_all_rows(data):
    """Convert all rows to float or np.nan."""
    new_data = []
    for row in data:
        new_data.append(convert_row_to_float(row))
    return np.array(new_data)

def nb_of_nans(data):
    nb_nans = np.zeros(data[0].shape)
    for i, col in enumerate(data.T):
        nb_nans[i] = np.count_nonzero(np.isnan(col))
    return nb_nans

def train_validation_split(data, ratio, seed):
    """Split data into training and validation set."""
    np.random.seed(seed)
    np.random.shuffle(data)
    # warning: if we shuffle our x_train, then we loose the corresponding y_train 
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

def removing_features(number_NaN,list_,data, threshold):
    Removed_features=[]
    for i in range(len(number_NaN)):
        if number_NaN[i] > round(len(data))*threshold:
            Removed_features.append(i)
    reduced_data = np.delete(data, Removed_features, 1)
    reduced_list = list(filter(lambda x: list_.index(x) not in Removed_features, list_))
    return reduced_data, reduced_list, Removed_features

def remove_identic_col(list_ids, x):
    """remove the column that have the same value everywhere to avoid std = 0 (also those columns are not useful)"""
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
    mean=np.nanmean(column)
    for i in range(len(column)):
        if np.isnan(column[i]):
            column[i]=mean
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

def replace_nans(data, method='mean'):
    for i in range(data.shape[1]):
        if method == 'mean':
            data[:, i] = np.nan_to_num(data[:, i], np.nanmean(data[:,i]))
        elif method == 'median':
            data[:, i] = np.nan_to_num(data[:, i], np.nanmedian(data[:,i]))
    return data

def create_dictionary_from_correlation(data, features ,correlation_threshold):
    newfeature_correlation_dict = {}
    # For each feature in the dataset calculate the correlation with the others and save those which have higher than 0.6 correlation
    for ft_num, feature in enumerate(features):
        newfeature_correlation_dict[feature] = []
        for o_ft_num, other_feature in enumerate(features):
            if (feature != other_feature):
                if np.abs(np.corrcoef(data[:,ft_num],data[:,o_ft_num])[0,1]) >= correlation_threshold:
                    newfeature_correlation_dict[feature].append(other_feature)
        print(f" Finished for feature: {feature}")
    return newfeature_correlation_dict

def replace_nine_with_nan(data):
    for i in range(data.shape[1]):
        if len(np.unique(data[:,i])) <= 9:
            for j in range(data.shape[0]):
                if data[j,i] == 9:
                    data[j,i] = np.nan

def replace_seven_with_nan(data):
    for i in range(data.shape[1]):
        if len(np.unique(data[:,i])) <= 9:
            if (7 in np.unique(data[:,i])) and (6 not in np.unique(data[:,i])):
                for j in range(data.shape[0]):
                    if data[j,i] == 7:
                        data[j,i] = np.nan

def replace_99_with_nan(data):
    for i in range(data.shape[1]):
        if len(np.unique(data[:,i])) <= 50:
            for j in range(data.shape[0]):
                if data[j,i] == 99 or data[j,i] == 77:
                    data[j,i] = np.nan
                if data[j,i] == 88:
                    data[j,i] = 0

def clean_outliers_modified(X):
    for column in range(X.shape[1]):  # for each feature:
        #For features that have more than 70 unique values
        if len(np.unique(column)) >= 70:
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
            # inside = np.invert(outside)
            #For values that are outside the bounds, replace them with NaN
            X[outside, column] = np.nan
    return X

def nan_corrcoef(data):
    """
    Calculate the correlation matrix between features, ignoring NaN values.

    Parameters:
    data (numpy.ndarray): The input data matrix.

    Returns:
    numpy.ndarray: The correlation matrix with NaN values appropriately handled.
    """
    num_features = data.shape[1]
    corr_matrix = np.empty((num_features, num_features), dtype=float)
    
    for i in range(num_features):
        for j in range(num_features):
            # Find indices where both columns have non-NaN values
            valid_indices = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
            
            # Calculate the correlation for valid values
            if np.any(valid_indices):
                corr_matrix[i, j] = np.corrcoef(data[valid_indices, i], data[valid_indices, j])[0, 1]
            else:
                # If no valid values, set the correlation to NaN
                corr_matrix[i, j] = np.nan

    return corr_matrix
