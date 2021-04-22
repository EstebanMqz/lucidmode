
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

# ----------------------------------------------------------------------------- READ PRE-LOADED DATASETS -- #
# --------------------------------------------------------------------------------------------------------- #

def dataset(p_dataset):
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

    if p_dataset == 'eth_ohlcv_H8':
    
        # read file from files folder
        df_data = pd.read_csv('files/prices/ETH_USDT_8h.csv')

    elif p_dataset == 'btc_ohlcv_H8':
        # read file from files folder
        df_data = pd.read_csv('files/prices/BTC_USDT_8h.csv')

    else:
        print('Error in: p_dataset')

    return df_data
