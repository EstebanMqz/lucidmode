
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- functions.py: python script with math functions                                                     -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# --------------------------------------------------------------------------------------- COST FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def cost(A, Y, type):

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
    elif type == 'multiclass-logloss':

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
    return cost
 
# --------------------------------------------------------------------------------- ACTIVATION FUNCTIONS -- #

def sigma(Z, activation):

    # -- Sigmoidal (sigmoid)
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    
    # -- Hyperbolic Tangent (tanh)
    elif activation == 'tanh':
        return np.tanh(Z)

    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)    
        return A
    
    # -- Softmax
    elif activation == 'softmax':
        expZ = np.exp(Z - np.max(Z)).T 
        return (expZ / expZ.sum(axis=0, keepdims=True)).T 

# ------------------------------------------------------------------- DERIVATIVE OF ACTIVATION FUNCTIONS -- #

def d_sigma(Z, activation):
    
    # -- Sigmoid
    if activation == 'sigmoid':
        s = sigma(Z, 'sigmoid')
        dZ = s*(1-s)
        assert (dZ.shape == Z.shape)
    
    # -- Hyperbolic Tangent
    elif activation == 'tanh':
        a = sigma(Z, activation)
        dZ = 1 - a**2
        assert (dZ.shape == Z.shape)
    
    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        dZ = np.array(Z, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
    
    # -- Softmax
    elif activation == 'softmax':
        s = sigma(Z, activation)
        s = s.reshape(-1, 1)
        dZ = np.diagflat(s) - np.dot(s, s.T)
        assert (dZ.shape == Z.shape)
    
    return dZ
