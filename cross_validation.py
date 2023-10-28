import numpy as np
from helpers import *
from implementations import reg_logistic_regression

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x,initial_w, max_iters, gamma, k_indices, k, lambda_, degree):
    """return the f1-scores of penalized logistic regression for a fold corresponding to k_indices"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    w0 = np.zeros(tx_tr.shape[1])
    # penalized logistic regression
    w, loss_tr = reg_logistic_regression(y_tr, tx_tr, lambda_,w0, max_iters, gamma)
    # calculate the F1-score for both training and test data
    pred_y_tr = tx_tr @ w
    pred_y_te = tx_te @ w
    F1_tr = compute_f1(y_tr,pred_y_tr)
    F1_te = compute_f1(y_te,pred_y_te)
    return F1_tr, F1_te