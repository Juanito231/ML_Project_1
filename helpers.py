import numpy as np

def compute_loss_mse(y, tx, w): #checked
    """Calculate the loss using mse."""
    e = y - tx @ w
    N = len(y)
    return  (e @ e.T/(2*N))

def sigmoid_function(z):
    """Apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-z))

def convert_predict(y):
    """ convert prediction to 0 and 1 with sigmoid"""
    return np.round(sigmoid_function(y))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w): #checked with gd
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_gradient_logistic(y, tx, w):#checked (lab5)
    """compute the gradient of loss.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    pred = sigmoid_function(tx.dot(w))
    grad = tx.T.dot(pred - y)/len(y)
    return grad

def compute_loss_logistic(y, tx, w): #checked
    """compute the cost by negative log likelihood."""
    # can simplify the loss as follows
    # loss = -ylog(sigma(x.T w)) - (1-y) log(1- sigma) = log(1+exp(x.T w)) - yx.Tw
    loss = np.log(1 + np.exp(tx @ w)) - y * (tx @ w)
    return np.mean(loss)

def regularized_logistic_loss(y, tx, w, lambda_):#checked, 
    #for some reason the tests pass when we DON'T ADD the regularized term
    loss = compute_loss_logistic(y, tx, w) #+ lambda_ * np.squeeze(w.T.dot(w))
    return loss

def regularized_logistic_gradient(y, tx, w, lambda_):
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return grad

def convert_minus1_to_0(y):
    return (y+1)/2 # -1 -> 0, 1 stays 1

def convert_0_to_minus1(y):
    return 2*y - 1

def convert_to_0_1(array_y):
    #If the value is greater than 0.5, we round it to 1, otherwise we round it to 0
    return map(array_y, lambda x: 1 if x >= 0.5 else 0)

def compute_accuracy(y, y_pred):
    """Computes the accuracy of the prediction.

    Args:
        y: shape=(N, ). The true labels.
        y_pred: shape=(N, ). The predicted labels.

    Returns:
        A scalar between 0 and 1, representing the fraction of correct predictions.
    """
    return np.mean(y == y_pred)

def compute_accuracy_logistic(y, tx, w):
    """Computes the accuracy of the prediction.

    Args:
        y: shape=(N, ). The true labels.
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        A scalar between 0 and 1, representing the fraction of correct predictions.
    """
    y_pred = convert_predict(tx @ w)
    return compute_accuracy(y, y_pred)

def compute_precision(y, y_pred):
    """Computes the precision of the prediction.

    Args:
        y: shape=(N, ). The true labels.
        y_pred: shape=(N, ). The predicted labels.

    Returns:
        A scalar between 0 and 1, representing the fraction of correct predictions.
    """
    true_positives = np.sum(np.logical_and(y == 1, y_pred == 1))
    false_positives = np.sum(np.logical_and(y == 0, y_pred == 1))
    return true_positives / (true_positives + false_positives)

def compute_recall(y, y_pred):
    """Computes the recall of the prediction.

    Args:
        y: shape=(N, ). The true labels.
        y_pred: shape=(N, ). The predicted labels.

    Returns:
        A scalar between 0 and 1, representing the fraction of correct predictions.
    """
    true_positives = np.sum(np.logical_and(y == 1, y_pred == 1))
    false_negatives = np.sum(np.logical_and(y == 1, y_pred == 0))
    return true_positives / (true_positives + false_negatives)

def compute_f1(y, y_pred):
    """Computes the f1 score of the prediction.

    Args:
        y: shape=(N, ). The true labels.
        y_pred: shape=(N, ). The predicted labels.

    Returns:
        A scalar between 0 and 1, representing the fraction of correct predictions.
    """
    precision = compute_precision(y, y_pred)
    recall = compute_recall(y, y_pred)
    return 2 * precision * recall / (precision + recall)

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
    pred_y_tr = convert_0_to_minus1(convert_predict(tx_tr @ w))
    pred_y_te = convert_0_to_minus1(convert_predict(tx_te @ w))
    F1_tr = compute_f1(y_tr,pred_y_tr)
    F1_te = compute_f1(y_te,pred_y_te)
    return F1_tr, F1_te