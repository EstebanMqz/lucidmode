
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
import pandas as pd
import os
import gzip

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

    # Base directory
    basedir = 'datasets/'
    
    # ------------------------------------------------------------------------------------ GENERIC MNIST -- #
    
    def load_mnist(path, kind='train'):
        """
        Helper function to read data for MNIST datasets
        """

        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images, labels

    # ------------------------------------------------------------------------------------ FASHION MNIST -- #

    if p_dataset == 'fashion-mnist':
        """
        28x28 pixel pictures of fashion clothes: https://github.com/zalandoresearch/fashion-mnist
        """

        folder = basedir + 'images/' + p_dataset + '/'
        X_train, y_train = load_mnist(folder, kind='train')
        X_test, y_test = load_mnist(folder, kind='t10k')

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    
    # ------------------------------------------------------------------------------------ DIGITS MNIST -- #

    elif p_dataset == 'digits-mnist':

        """
        28x28 pixel pictures of handwritten digits: http://yann.lecun.com/exdb/mnist/
        """

        folder = basedir + 'images/' + p_dataset + '/'
        X_train, y_train = load_mnist(folder, kind='train')
        X_test, y_test = load_mnist(folder, kind='t10k')

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    
     # ---------------------------------------------------------------------------------- GENETIC FINANCE -- #

    elif p_dataset == 'genetic-finance':
    
        folder = basedir + 'timeseries/' + p_dataset + '/'
        
        # read file from files folder
        X_train = pd.read_csv(folder + 'X_train.csv').iloc[:, 2:]
        y_train = pd.read_csv(folder + 'y_train.csv')
        X_val = pd.read_csv(folder + 'X_val.csv').iloc[:, 2:]
        y_val = pd.read_csv(folder + 'y_val.csv')
        
        return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
        
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
