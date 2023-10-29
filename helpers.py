import numpy as np

def compute_loss_mse(y, tx, w): 
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

def prediction(tx, w):
    """ return the prediction based on the data tx and the estimate w from the model"""
    return convert_predict(tx@w)

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

def compute_gradient(y, tx, w): 
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

def compute_gradient_logistic(y, tx, w):
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

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    # we simplified the loss as follows
    # loss = -ylog(sigma(x.T w)) - (1-y) log(1- sigma) = log(1+exp(x.T w)) - yx.Tw
    loss = np.log(1 + np.exp(tx @ w)) - y * (tx @ w)
    return np.mean(loss)

def regularized_logistic_loss(y, tx, w, lambda_):
    # it was specified that we don't add the regularized term
    loss = compute_loss_logistic(y, tx, w) #+ lambda_ * np.squeeze(w.T.dot(w))
    return loss

def regularized_logistic_gradient(y, tx, w, lambda_):
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return grad

def convert_minus1_to_0(y):
    return (y+1)/2 

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
    true_positives = np.nansum(np.logical_and(y == 1, y_pred == 1))
    false_positives = np.nansum(np.logical_and(y == 0, y_pred == 1))
    if (true_positives + false_positives) == 0:
        return 0.0
    else:
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
    if (true_positives + false_negatives) == 0:
        return 0.0
    else:
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
    if (precision + recall) == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)