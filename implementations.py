# Implementations of ML methods -- Step 2
from helpers import*
import numpy as np

# all functions should return: (w, loss), which is the last weight vector of the
#method, and the corresponding loss value (cost function).

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma): #checked
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    # initialize w
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
    """#if we want the details of the convergence
        loss = compute_loss_mse(y, tx, w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    """
    # compute the final loss
    loss = compute_loss_mse(y, tx, w)
    return w, loss
    

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma): #checked
    """Linear regression using stochastic gradient descent

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    w = initial_w
    for n_iter in range(max_iters):
        # we use a batch_size of 1 (as stated in the project description)
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
        """ if we want the details of the convergence, uncomment this part
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l= compute_loss_mse(y, tx, w), w0=w[0], w1=w[1]))
        """
    # compute the final loss
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):#checked
    """Least squares regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,D)
    """
    #solve linear system with np.linalg.solve
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss_mse(y,tx,w)
    return w, loss


def ridge_regression(y, tx, lambda_): #checked
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape=(N,D)
        lambda_: regularization parameter
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss_mse(y,tx,w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):#checked
    """Logistic regression using GD (y in {0,1})

    Args:
        y: 
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient_logistic(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
        """#if want details of the convergence
        loss = compute_loss_logistic(y, tx , w)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
              """
        
    #compute final loss
    loss = compute_loss_logistic(y, tx, w)
    return w, loss
   
def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using SGD (y in {0,1})

    Args:
        y: 
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
    """
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient_logistic(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
        """    
        loss = compute_loss_logistic(y, tx, w)
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        """
    #compute final loss
    loss = compute_loss_logistic(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma ):#checked
    """Regularized logistic regression using GD (y in {0,1},
    regularization term lambda*||w||^2)

    Args:
        y: 
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda_: regularization parameter
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad = regularized_logistic_gradient(y, tx, w, lambda_)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        """#print details
        loss = regularized_logistic_loss(y, tx , w, lambda_)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    """
    #compute final loss
    loss = regularized_logistic_loss(y, tx, w, lambda_)
    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using SGD (y in {0,1},
    regularization term lambda*||w||^2)

    Args:
        y: 
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). Initial guess for our model
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda_: regularization parameter
    """
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad = regularized_logistic_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad    
        """print details
        # calculate loss
        loss = regularized_logistic_loss(y, tx, w)
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        """        
    loss = regularized_logistic_loss(y, tx, w)
    return w, loss