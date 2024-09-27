'''
Test Cases
--------
>>> y_N = 0.0
>>> yhat_N = 4.123
>>> calc_root_mean_squared_error(y_N, yhat_N)
4.123

>>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
>>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
>>> rmse = calc_root_mean_squared_error(y_N, yhat_N)
>>> np.round(rmse, 6)
1.154701
'''

import numpy as np
import math

def calc_root_mean_squared_error(y_N, yhat_N):
    ''' Compute root mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    rmse : scalar float
        Root mean squared error performance metric
        .. math:
            rmse(y,\hat{y}) = \sqrt{\frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2}
    '''
    #input handling to ensure 1d
    y_N = np.atleast_1d(y_N)
    yhat_N = np.atleast_1d(yhat_N)
    #assertions to ensure actual value (y_N) and predicted value (yhat_N) are the same shape and 1-dimensional array
    assert y_N.ndim == 1
    assert y_N.shape == yhat_N.shape
    formula =  np.round(math.sqrt(np.sum((y_N - yhat_N) ** 2) / y_N.shape[0]), 6)
    return formula

#Test Case 1
y_N = np.array([1.0, 2.0, 3.0])
yhat_N = np.array([1.0, 2.0, 3.0])
print(calc_root_mean_squared_error(y_N, yhat_N))  # Expected output: 0.0

#Test Case 2
y_N = np.array([1.0, 2.0, 3.0])
yhat_N = np.array([2.0, 3.0, 4.0])
print(calc_root_mean_squared_error(y_N, yhat_N))  # Expected output: 1.0

#Test Case 3
y_N = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
yhat_N = np.array([1.5, 1.8, 2.5, 4.2, 4.8])
print(calc_root_mean_squared_error(y_N, yhat_N))  # Expected output: Non-zero value, roughly 0.35

#Test Case 4
y_N = 0.0
yhat_N = 4.123
print(calc_root_mean_squared_error(y_N, yhat_N)) #4.123

#Test Case 5
y_N = np.asarray([-2, 0, 2])
yhat_N = np.asarray([-4, 0, 2])
print(calc_root_mean_squared_error(y_N, yhat_N)) #1.154701