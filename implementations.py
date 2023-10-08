# Implementations of ML methods -- Step 2
from helpers import*
import numpy as np

# all functions should return: (w, loss), which is the last weight vector of the
#method, and the corresponding loss value (cost function).

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    raise NotImplementedError
    

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
    """
    #solve linear system with np.linalg.solve
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,2)
        lambda_: regularization parameter
    """
    coeff = lambda_*2*len(y) # lambda'
    # N = len(y), D = len(tx[0])
    w = np.linalg.solve(tx.T @ tx + coeff * np.eye(len(tx[0])), tx.T @ y)
    mse = compute_loss(y, tx, w)
    return w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using GD or SGD (y in {0,1})

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD or SGD (y in {0,1},
    regularization term lambda*||w||^2)

    Args:
        y: 
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    raise NotImplementedError