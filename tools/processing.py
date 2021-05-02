
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- processing.py: python script with data pre-post processing functions                                -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ------------------------------------------------------------------------------------- TRAIN_TEST_SPLIT -- #
# --------------------------------------------------------------------------------------------------------- #

def train_val_split(x_data, y_data, train_size=0.8, random_state=1):
    """
    
    To split into train and validation split with an optional third split for final test.

    """

    np.random.seed(random_state)
    arr_rand = np.random.rand(x_data.shape[0])
    split = arr_rand < np.percentile(arr_rand, train_size*100)

    x_train = x_data[split]
    y_train = y_data[split]
    x_val =  x_data[~split]
    y_val = y_data[~split]

    return x_train, x_val, y_train, y_val
