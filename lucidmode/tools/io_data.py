
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

    # ---------------------------------------------------------------------------------- GENETIC FINANCE -- #

    if p_dataset == 'genetic_finance':
    
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
    
    # ------------------------------------------------------------------------------------ FASHION MNIST -- #

    elif p_dataset == 'fashion_MNIST':
        """
        28x28 pixel pictures of fashion clothes: https://github.com/zalandoresearch/fashion-mnist
        """

        folder = basedir + 'images/' + p_dataset + '/'
        file_1 = 'train-labels-idx1-ubyte'
        file_2 = 'train-images-idx3-ubyte'

        # -- read files from local system folder (both )
        labels_path = os.path.join(folder + file_1)
        with open(labels_path,'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
                
        images_path = os.path.join(folder + file_2)
        with open(images_path,'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        # SIMPLIER VERSION: drop all samples from the following class. 
        """
        todrop = [4, 5, 6, 7, 8, 9]
        for i in todrop:
            idxs = (labels == i)
            images = images[~idxs]
            labels = labels[~idxs]
        """

        return {'images': images, 'labels': labels}
    
    # ------------------------------------------------------------------------------------ DIGITS MNIST -- #

    elif p_dataset == 'digits_MNIST':

        """
        28x28 pixel pictures of handwritten digits: http://yann.lecun.com/exdb/mnist/
        """

        # -- read files from local system folder (both )
        labels_path = os.path.join('datasets/images/' + p_dataset + '/', 'train-labels-idx1-ubyte')
        with open(labels_path,'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).reshape(-1, 1)
                
        images_path = os.path.join('files/data/images/' + p_dataset + '/', 'train-images-idx3-ubyte')
        with open(images_path,'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        
        # normalize images 
        img_norm = images / 255

        return {'images': img_norm, 'labels': labels}

    else:
        print('Error in: p_dataset')
