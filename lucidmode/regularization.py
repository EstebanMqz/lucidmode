
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- visualizations.py: python script with visualization functions                                       -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ---------------------------------------------------------------------- L1-L2-ELASTICNET REGULARIZATION -- #
# --------------------------------------------------------------------------------------------------------- #

def _l1_l2_EN(n_weights, type=None, lmbda=0, ratio=0):
    """
    Regularization operation, can perform 3 types, L1 (Lasso), L2 (Ridge), Elastic Net = ratio(L1 - L2)

    In Cost Function:
    - Weights values of all layers (l1, l2, Elastic Net)

    In weights updating process of selected layers:
    - Applied to weights values (l1, l2, Elastic Net)
    - Applied to bias values (l1, l2, Elastic Net)

    Parameters
    ----------

    n_weights: list of np.array
        
        Containing the weights values to be regularized

    type: {'l1', 'l2', 'elasticnet'}, default='l1' 
        
        'l1': Lasso --> (lambda) * ||weights||_1
        'l2': Ridge --> (lambda) * 0.5 * ||weights||^2_2
        'elasticnet': ratio * L1 + (1 - ratio) * L2

    lmbda: float
        
        value for lambda term in the regularization, between 0 and 1

    ratio: float
        
        Used with the option type='elasticnet', is the ratio between l1 and l2 impact on the regularization

    Return
    ------
    
    reg_value: float
        
        The result of the calculation squeezed to shape (1,)


    References
    ----------

    [3] Zou and Hastie. Regularization and variable selection via the elastic net. Journal Of The Royal     
        Statistical Society 67(2):301 - 320, 2005.

    """

    if type == None:
        reg = 0

    elif type == 'l1':
        reg = 0
        for weights in n_weights:
            reg += (lmbda) * np.sum(abs(weights))

    elif type == 'l2':
        reg = 0
        for weights in n_weights:
            reg += (lmbda) * 0.5 * np.sum(weights**2)

    elif type == 'elasticnet':
        reg = 0
        for weights in n_weights:
            reg += ratio*((lmbda) * np.sum(abs(weights))) + (1 - ratio)*((lmbda) * 0.5 * np.sum(weights**2))
    
    return reg.astype(np.float32)

# ------------------------------------------------------------------------------- DROPOUT REGULARIZATION -- #
# --------------------------------------------------------------------------------------------------------- #

# - In layers
# - Neurons activation (dropout)
