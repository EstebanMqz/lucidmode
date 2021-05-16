
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- data.py: python script with data input/output and processing tools                                  -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ----------------------------------------------------------------------------------- READ DATASET FILES -- #
# --------------------------------------------------------------------------------------------------------- #

def datasets(p_dataset):
    """
    Read internally generated datasets for exploration purposes.

    Parameters
    ----------
    
    p_dataset:
    
    Returns
    -------

    References
    ----------

    """
   
    # --------------------------------------------------------------------------------------- RANDOM XOR -- #
    
    if p_dataset == 'xor':
        
        # generate random data 
        np.random.seed(1)
        x = np.random.randn(200, 2)
        y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
        y = y.reshape(y.shape[0], 1)
        
        return {'y': y, 'x': x}
    
    else:
        print('Error in: p_dataset')
