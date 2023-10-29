"""Utility functions for data preprocessing."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def train_validation_split(data, ratio, seed):
    """Split data into training and validation sets."""
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

def removing_features(number_NaN,list_,data, threshold):
    """Remove features that have more than 100*threshold % of nans."""
    Removed_features=[]
    for i in range(len(number_NaN)):
        if number_NaN[i] > round(len(data))*threshold:
            Removed_features.append(i)
    reduced_data = np.delete(data, Removed_features, 1)
    reduced_list = list(filter(lambda x: list_.index(x) not in Removed_features, list_))
    return reduced_data, reduced_list, Removed_features

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

def clean_outliers(X):
    """Clean the outliers in each column."""
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
