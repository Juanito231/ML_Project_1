# Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import seaborn as sns
from preprocess import *
from helpers_given import *
from implementations import *
from cross_validation import *

# Load data and convert to float
path = "dataset"
x_train, x_test, y_train, train_ids, test_ids =  load_csv_data(path, sub_sample=False)
with open(path + '/x_train.csv', 'r') as f:
    features_string = f.readline()
    features = features_string.split(',')
features = features[1:]
#data = x_train
# Features to keep:
feature_indeces = [26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46,
                   47, 48, 50, 51, 52, 53, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71,
                   72, 76, 87, 99, 100, 103, 104, 216, 217, 219, 220, 221, 222, 227, 229, 
                   230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 244, 245, 247,
                   248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 264, 266, 267,
                   268, 269, 270, 271, 276, 277, 280, 281, 287, 288, 297, 313, 314, 315, 320]

NaNs = nb_of_nans(x_train)
# convert y from {-1,1} to {0,1} to avoid problems with logistic
y_01 = convert_minus1_to_0(y_train)

reduced_x_train = x_train[:, feature_indeces]
reduced_x_test = x_test[:, feature_indeces]
# Replace nine values with NaNs
replace_nine_with_nan(reduced_x_train)
replace_nine_with_nan(reduced_x_test)

# Replace seven values with NaNs
replace_seven_with_nan(reduced_x_train)
replace_seven_with_nan(reduced_x_test)

# Replace 99 values with NaNs
replace_99_with_nan(reduced_x_train)
replace_99_with_nan(reduced_x_test)

# Remove outliers
reduced_x_train = clean_outliers_modified(reduced_x_train)
reduced_x_test = clean_outliers_modified(reduced_x_test)

# Replace NaNs with medians
reduced_median = replace_NaN(reduced_x_train, method='median')
reduced_test_median = replace_NaN(reduced_x_test, method = 'median')

# Standardize the data
standardized_x_train = standardize_data(reduced_median)
standardized_x_test = standardize_data(reduced_test_median)

best_degree = 1
best_lambda = 0.5

tx_tr = build_poly(standardized_x_train, 1)
tx_te = build_poly(standardized_x_test, 1)

w_reg, loss_reg = reg_logistic_regression(y_01, tx_tr, initial_w = np.zeros(tx_tr.shape[1]), max_iters = 100, gamma = 0.5, lambda_ = best_lambda)
# print(loss_reg)
y_pred = convert_0_to_minus1(prediction(tx_te, w_reg))

# create submission file
create_csv_submission(test_ids, y_pred, "best_result")
