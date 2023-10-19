import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def sigmoid_function(z):
    """Apply sigmoid function on t."""
    return 1 / (1 + np.exp(-z))

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
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid_function(tx.dot(w))
    grad = (1/len(y))* tx.T.dot(pred - y)
    return grad

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid_function(tx.dot(w))
    # We know that P(y=0|x) = 1 - P(y=1|x)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return -(1/len(y)) * np.squeeze(- loss)

def regularized_logistic_loss(y, tx, w, lambda_):
    loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss

def regularized_logistic_gradient(y, tx, w, lambda_):
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return grad
