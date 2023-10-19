# Implementations of ML methods -- Step 2
from helpers import*
import numpy as np

# all functions should return: (w, loss), which is the last weight vector of the
#method, and the corresponding loss value (cost function).

# change so we don't keep all weights and losses in memory (bc big dataset)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    weights, losses = gradient_descent(y, tx, initial_w, max_iters, gamma)
    w = weights[-1]
    loss = losses[-1]
    return w, loss
    

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    weights, losses = stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)
    w = weights[-1]
    loss = losses[-1]
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
    """
    #solve linear system with np.linalg.solve
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y,tx,w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        lambda_: regularization parameter
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return w, loss

def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using GD (y in {0,1})

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    weights, losses = GD_logistic(y, tx, initial_w, max_iters, gamma)
    w = weights[-1]
    loss = losses[-1]
    return w, loss
   
def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Logistic regression using SGD (y in {0,1})

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
    """
    weights, losses = SGD_logistic(y, tx, initial_w, batch_size, max_iters, gamma)
    w = weights[-1]
    loss = losses[-1]
    return w, loss

def regularized_logistic_regression_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Regularized logistic regression using GD (y in {0,1},
    regularization term lambda*||w||^2)

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda_: regularization parameter
    """
    weights, losses = Regularized_GD_logistic(y, tx, initial_w, max_iters, gamma, lambda_)
    w = weights[-1]
    loss = losses[-1]
    return w, loss

def regularized_logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma, lambda_):
    """Regularized logistic regression using SGD (y in {0,1},
    regularization term lambda*||w||^2)

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda_: regularization parameter
    """
    weights, losses = Regularized_SGD_logistic(y, tx, initial_w, batch_size, max_iters, gamma, lambda_)
    w = weights[-1]
    loss = losses[-1]
    return w, loss