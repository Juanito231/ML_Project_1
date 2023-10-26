# Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import seaborn as sns
from preprocess import *
from helpers_given import *
from implementations import *

# Load data and convert to float
x_train, x_test, y_train, train_ids, test_ids =  load_csv_data('dataset', sub_sample=False)
with open('dataset/x_train.csv', 'r') as f:
    features_string = f.readline()
    features = features_string.split(',')
features = features[1:]
data = x_train
# data = convert_all_rows(x_train)
# NaNs = number_of_NaNs(features,data)
# faster alternative for number of nans:
NaNs = nb_of_nans(data)
# convert y from {-1,1} to {0,1} to avoid problems with logistic
y_01 = convert_minus1_to_0(y_train)

# Get reduced data
reduced_data, reduced_features, Removed_features = removing_features(NaNs,features,data)
# remove the same columns from x_test
reduced_x_test = np.delete(x_test, Removed_features, 1)

# Get feature correlation dictionary
feature_correlation_dict = create_dictionary_from_correlation(reduced_data,reduced_features,0.6)

max_corrr_feature_dict = {}

# Find the 50 most correlated features
for key, val in feature_correlation_dict.items():
    max_corrr_feature_dict[key] = len(val)

# Sort the dictionary by value in descending order
max_corrr_feature_dict = {k: v for k, v in sorted(max_corrr_feature_dict.items(), key=lambda item: item[1], reverse=True)}

features_to_drop = []
for key in max_corrr_feature_dict.keys():
    features_to_drop.append(key)
    if len(features_to_drop) == 30:
        break

# Define features to keep
features_to_keep = []
for feature in reduced_features:
    if feature not in features_to_drop:
        features_to_keep.append(feature)

# Also replace some features with their calculated counterparts
origin_calculated_features = {
    'WEIGHT2' : 'WTKG3',
    'HEIGHT3' : 'HTM4',
    'ALCDAY5' : '_DRNKWEK',
    'FRUITJU1' : 'FTJUDA1_',
    'FRUIT1' : 'FRUTDA1_',
    'FVBEANS' : 'BEANDAY_',
    'FVGREEN' : 'GRENDAY_',
    'FVORANG' : 'ORNGDAY_',
    'VEGETAB1' : 'VEGEDA1_',
    'STRENGTH' : 'STRFREQ_'
}

# In features_to_keep replace the key of origin_calculated_features with the value
for key, val in origin_calculated_features.items():
    for i, feature in enumerate(features_to_keep):
        if key == feature:
            features_to_keep[i] = val

# Drop duplicates
features_to_keep = list(set(features_to_keep))

# Get the indices of the selected features
selected_features_indices = []
for feature in features_to_keep:
    selected_features_indices.append(reduced_features.index(feature))

selected_features_indices = sorted(selected_features_indices)

# Create a new dataset with keeping the features that are in the selected_features_indices
reduced_data = reduced_data[:, selected_features_indices]
reduced_x_test = reduced_x_test[:, selected_features_indices]

# Also remove the features from the reduced_features list
reduced_features_2 = []
for feature in reduced_features:
    if feature in features_to_keep:
        reduced_features_2.append(feature)

# Remove redundant features
redundant_features = [ 'FMONTH','IDATE','IMONTH','IDAY','IYEAR', 'SEQNO', '_STATE', '_PSU', ]
# Get the indices of these features
redundant_features_indices = []
for feature in redundant_features:
    redundant_features_indices.append(reduced_features_2.index(feature))

# Create a new dataset with removing the features that are in the selected_features_indices
reduced_data = np.delete(reduced_data, redundant_features_indices, 1)
reduced_features_2 = [reduced_features_2[i] for i in range(len(reduced_features_2)) if i not in redundant_features_indices]
reduced_x_test = np.delete(reduced_x_test, redundant_features_indices, 1)

# Replace nine values with NaNs
replace_nine_with_nan(reduced_data)
replace_nine_with_nan(reduced_x_test)

# Replace seven values with NaNs
replace_seven_with_nan(reduced_data)
replace_seven_with_nan(reduced_x_test)

# Replace 99 values with NaNs
replace_99_with_nan(reduced_data)
replace_99_with_nan(reduced_x_test)

# Remove outliers
reduced_data = clean_outliers_modified(reduced_data)
# should we do this for x_test too ?

# For the _DRNKWEK feature, replace 9990 with NaN
# TODO: maybe we could vectorize this so we don't do a loop over the data
for i in range(reduced_data.shape[0]): 
    if reduced_data[i, reduced_features_2.index('_DRNKWEK')] == 9990:
        reduced_data[i, reduced_features_2.index('_DRNKWEK')] = np.nan
for i in range(reduced_x_test.shape[0]): 
    if reduced_x_test[i, reduced_features_2.index('_DRNKWEK')] == 9990:
        reduced_x_test[i, reduced_features_2.index('_DRNKWEK')] = np.nan

# Replace NaNs with medians
reduced_median = replace_NaN(reduced_data, method='median')
reduced_test_median = replace_NaN(reduced_x_test, method = 'median')

# Standardize the data
standardized_x = standardize_data(reduced_median)
standardized_test = standardize_data(reduced_test_median)

# Run the regularized logistic 
NB_COL = standardized_x.shape[1] # corresponds to 'D' = number of features
NB_ROWS = standardized_x.shape[0] # corresponds to 'N' = number of observations/respondents
w_reg, loss_reg = reg_logistic_regression(y_train, standardized_x, initial_w = np.zeros(NB_COL), max_iters = 50, gamma = 0.1, lambda_ = 0.5)
# print(loss_reg)
y_pred = convert_0_to_minus1(convert_predict(standardized_test @ w_reg))

# create submission file
create_csv_submission(test_ids, y_pred, "submission")