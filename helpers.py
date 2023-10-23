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
