
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

import numpy as np

# --------------------------------------------------------------------------------- ACTIVATION FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

def _sigma(z, activation):
    """

    """

    # -- Sigmoid
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
        
    # -- Tanh
    elif activation == 'tanh':
        return np.tanh(z)
    
    # -- ReLu
    elif activation == 'relu':
        return np.tanh(z)
    
    # -- Linear
    elif activation == 'linear':
        return 1
    
    # -- Softmax
    elif activation == 'softmax':
        return 2


# --------------------------------------------------------------------------------------- COST FUNCTIONS -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Binary Cross-Entropy 
# -- Multiclass Cross-Entropy
# -- Mean Squared Error
