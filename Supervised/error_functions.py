import numpy as np

def residuals(y, y_hat):
    return y - y_hat

def square_residuals(y, y_hat):
    diff = residuals(y, y_hat)
    return np.square(diff)

def absolute_residuals(y, y_hat):
    diff = residuals(y, y_hat)
    return np.abs(diff)

def mean_squared_error(y, y_hat):
    N = len(y)
    return np.sum(square_residuals(y, y_hat)) / N

def mean_absolute_error(y, y_hat):
    N = len(y)
    return np.sum(absolute_residuals(y, y_hat)) / N

