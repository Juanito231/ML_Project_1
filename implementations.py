# Implementations of the 6 ML methods
from helpers import*
import numpy as np

# All functions return: (w, loss), which are the last weight vector of the
# method, and the corresponding loss value.

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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


def least_squares(y, tx):
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


def ridge_regression(y, tx, lambda_):
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
    previous_loss = 0
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient_logistic(y, tx, w)
        loss = compute_loss_logistic(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
        if (np.abs(loss - previous_loss) < 1e-8) and (n_iter > 1):
            break
        previous_loss = loss
        """#if want details of the convergence
        loss = compute_loss_logistic(y, tx , w)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
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
