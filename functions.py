
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

# -- basic
import numpy as np

# --------------------------------------------------------------------------------------- COST FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def _cost(A, Y, type):
    """
    """

    # -- Mean Squared Error
    if type == 'mse':
        
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
# --------------------------------------------------------------------------------------------------------- #

def __sigma(Z, activation):
    """

    """

    # -- Sigmoidal (sigmoid)
    if activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
    
    # -- Hyperbolic Tangent (tanh)
    elif activation == 'tanh':
        A = np.tanh(Z)

    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)    
    
    # -- Softmax
    elif activation == 'softmax':
        expZ = np.exp(Z - np.max(Z)).T 
        A = (expZ / expZ.sum(axis=0, keepdims=True)).T
    
    return A, Z

# ------------------------------------------------------------------- DERIVATIVE OF ACTIVATION FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def __dsigma(dA, Z, activation):
    """
    """

    # -- Sigmoid
    if activation == 'sigmoid':
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
    
    # -- Hyperbolic Tangent
    elif activation == 'tanh':
        a = __sigma(Z, activation)
        dZ = 1 - a**2
        assert (dZ.shape == Z.shape)
    
    # -- Rectified Linear Unit (ReLU)
    elif activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
    
    # -- Softmax
    elif activation == 'softmax':
        s = __sigma(Z, activation)
        s = s.reshape(-1,1)
        dZ = np.diagflat(s) - np.dot(s, s.T)
        assert (dZ.shape == Z.shape)
    
    return dZ


# --------------------------------------------------------------------------- FORWARD/BACKWARD FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def __s_forward(A, W, b):
    """
    """

    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def __forward(A_prev, W, b, activation):

    Z, linear_cache = __s_forward(A_prev, W, b)
    A, activation_cache = __sigma(Z, activation)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache 


def __s_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]
   
    dW = (1/m)*np.dot(dZ,np.transpose(A_prev))
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(np.transpose(W),dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def __backward(dA, cache):

    linear_cache, activation_cache = cache
    dZ =__dsigma(dA, activation_cache)
    dA_prev, dW, db = __s_backward(dZ, linear_cache)

    return dA_prev, dW, db


