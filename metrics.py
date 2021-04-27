
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- metrics.py: python script with a variaty of useful metrics                                          -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ----------------------------------------------------------------------------------- FOR CLASSIFICATION -- #
# --------------------------------------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------------- Accuracy -- # 

def accuracy(y, y_hat):
    """
    """

    # -- SINGLE-CLASS
    if len(list(np.unique(y))) == 2:

        tp = sum(y[y_hat == 1] == 1)
        fn = sum(y[y_hat == 0] == 1)
        fp = sum(y[y_hat == 1] == 0)
        tn = sum(y[y_hat == 0] == 0)
        
        return (tp + tn)/(fp + fn + tp + tn)
    
    # -- MULTI-CLASS
    else: 
        classes = len(np.unique(y))
        cm = np.zeros((classes, classes))

        for i in range(len(y)):
            cm[y[i]][y_hat[i]] += 1

        return np.sum(np.diag(cm))/y.shape[0]


# ------------------------------------------------------------------------------------ Confussion Matrix -- # 

def confussion_matrix(y, y_hat):

    k = 10
    mat = np.zeros((k, k)).astype(int)
    for i in range(k):
        T = (y_hat == i) #all equal to class k
        for j in range(k):
            mat[i, j] = sum(y[T] == j)

    return mat

# -------------------------------------------------------------------------------------------- Precision -- #

# ----------------------------------------------------------------------------------------------- Recall -- #

# -------------------------------------------------------------------------------------------------- AUC -- #


# --------------------------------------------------------------------------------------- FOR REGRESSION -- #
# --------------------------------------------------------------------------------------------------------- #

# -- R2

# ----------------------------------------------------------------------------------------- FOR LEARNING -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Kullback-Liebler
