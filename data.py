
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- data.py: python script with data input/output and processing tools                                  -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------- READ PRE-LOADED DATASETS -- #
# --------------------------------------------------------------------------------------------------------- #

def datasets(p_dataset):
    """
    Read different datasets, from publicly known like the MNIST series, to other particularly built
    for this project, like OHLCV cryptocurrencies prices Time series.

    
    Parameters
    ----------
    
    p_dataset:
    
    Returns
    -------

    References
    ----------

    """

    # ------------------------------------------------------------------------------------------- ETH H8 -- #

    if p_dataset == 'eth_ohlcv_H8':
    
        # read file from files folder
        return pd.read_csv('files/prices/ETH_USDT_8h.csv')

    # ------------------------------------------------------------------------------------------- BTC H8 -- #

    elif p_dataset == 'btc_ohlcv_H8':

        # read file from files folder
        return pd.read_csv('files/prices/BTC_USDT_8h.csv')
    
    
    # --------------------------------------------------------------------------------------- RANDOM XOR -- #
    
    elif p_dataset == 'xor':
        
        # generate random data 
        np.random.seed(1)
        x = np.random.randn(200, 2)
        y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
        y = y.reshape(y.shape[0], 1)
        
        return {'y': y, 'x': x}

    else:
        print('Error in: p_dataset')
