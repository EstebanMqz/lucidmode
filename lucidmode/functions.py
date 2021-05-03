
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- functions.py: python script with math functions                                                     -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# --------------------------------------------------------------------------------------- COST FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def _cost(A, Y, type):
    
    # numerical stability parameter
    ns = 1e-25
    A = A + ns

    # -- Mean Squared Error
    if type == 'sse':
        
        # loss as the difference on prediction
        loss = A - Y
        # cost as the sum of the squared errors
        cost = np.sum(((loss)**2))

    # -- Binary Cross-Entropy (pending)
    elif type == 'binary-logloss':
        
        # loss as the errors within each value
        loss = np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))
        # cost as the mean of loss
        cost = -(1/Y.shape[1]) * np.sum(loss)
    
    # -- Multiclass Cross-Entropy (pending)
    elif type == 'multi-logloss':

        # auxiliary object
        y_hat = np.zeros(shape=(Y.shape[0], A.shape[1]))
        y_hat[range(len(Y)), Y] = 1

        # loss as the errors within each value
        loss = np.sum(-y_hat * np.log(A))
        # cost as the mean of loss
        cost = np.sum(loss)/y_hat.shape[0]
    
    # compress to 0 dimension (to have a float)
    cost = np.squeeze(cost)

    # check final dimensions
    assert(cost.shape == ())

    # function final result
    return cost.astype(np.float32).round(decimals=4)
 
# --------------------------------------------------------------------------------- ACTIVATION FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def _sigma(Z, activation):

    # -- Sigmoidal (sigmoid)
    if activation == 'sigmoid':
        Z = 1 / (1 + np.exp(-Z))
        Z = Z.astype(np.float32)
        return Z
    
    # -- Hyperbolic Tangent (tanh)
    elif activation == 'tanh':
        Z = np.tanh(Z)
        Z = Z.astype(np.float32)
        return Z

    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)
        return A
    
    # -- Softmax
    elif activation == 'softmax':

        expZ = np.exp(Z - np.max(Z)).T 
        Z = (expZ / expZ.sum(axis=0, keepdims=True)).T 
        Z = Z.astype(np.float32)
        return Z

# ------------------------------------------------------------------- DERIVATIVE OF ACTIVATION FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def _dsigma(Z, activation):
    
    # -- Sigmoid
    if activation == 'sigmoid':
        s = _sigma(Z, activation)
        dZ = s*(1-s)
        dZ = dZ.astype(np.float32)
        assert (dZ.shape == Z.shape)
    
    # -- Hyperbolic Tangent
    elif activation == 'tanh':
        a = _sigma(Z, activation)
        dZ = 1 - a**2
        dZ = dZ.astype(np.float32)
        assert (dZ.shape == Z.shape)
    
    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        dZ = np.array(Z, copy=True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        dZ = dZ.astype(np.int8)
        assert (dZ.shape == Z.shape)
    
    # -- Softmax
    elif activation == 'softmax':
        s = _sigma(Z, activation)
        s = s.reshape(-1, 1)
        s = s.astype(np.float32)
        dZ = np.diagflat(s) - np.dot(s, s.T)
        assert (dZ.shape == Z.shape)
    
    return dZ
