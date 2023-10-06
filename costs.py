import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx.dot(w)
    return calculate_mse(e)
