
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- experiments.py: python script with experiment cases                                                 -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- load class
from lucidmode.models import NeuralNet

# -- base libraries
import numpy as np

# -- complementary tools
from rich import inspect
from lucidmode.tools.metrics import metrics
from lucidmode.tools.processing import train_val_split, gridsearch

# ------------------------------------------------------------------------------------- IMAGE CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

"""
This example is for an image classifier using mnist data sets
"""

# -------------------------------------------------------------------------------- LOAD DATA FROM FOLDER -- #

def _load_mnist(path, kind='train'):
        """
        Helper function to read data for MNIST datasets
        """

        import os
        import gzip

        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images, labels

path = 'lucidmode/datasets/images/fashion-mnist/'    
X_train, y_train = _load_mnist(path, kind='train')
X_test, y_test = _load_mnist(path, kind='t10k')

# split data
X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, train_size = 0.3, random_state = 1)

# --------------------------------------------------------------------------------------- MODEL FORMATION-- #

# Neural Net Topology Definition
lucid = NeuralNet(hidden_l=[60, 30, 10], hidden_a=['tanh', 'tanh', 'tanh'],
                  hidden_r=[{'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                            {'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                            {'type': 'l1', 'lmbda': 0.001, 'ratio':0.1}],
                
                  output_r={'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                  output_n=10, output_a='softmax')

# Model and implementation case Formation
lucid.formation(cost={'function': 'multi-logloss', 'reg': {'type': 'l1', 'lmbda': 0.001, 'ratio':0.1}},
                init={'input_shape': X_train.shape[1], 'init_layers': 'common-uniform'},
                optimizer={'type': 'SGD', 'params': {'learning_rate': 0.075, 'batch_size': 18000}},
                metrics=['acc'])

# Inspect object contents  (Weights initialization)
# inspect(lucid)

# ---------------------------------------------------------------------------------------- MODEL TRAINING-- #

# cost evolution
lucid.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, epochs=100, verbosity=3)

# acces to the train history information
history = lucid.history

# Predict train
y_hat = lucid.predict(x_train=X_train)
train_metrics = metrics(y_train, y_hat, type='classification')

# Confusion matrix
train_metrics['cm']

# Overall accuracy
train_metrics['acc']

# Predict train
y_val_hat = lucid.predict(x_train=X_val)
val_metrics = metrics(y_val, y_val_hat, type='classification')

# Overall accuracy
val_metrics['acc']

"""

- epoch: 100 
 ---------------------------------------  
- cost_train: 0.9596 - cost_val: 0.9433
- acc_train: 0.6904 - acc_val: 0.6910
 
>>> train_metrics['cm']
 array([[1413,   43,   37,  134,   24,    0,  103,    0,   38,    0],
       [  58, 1542,   34,   86,   16,    0,    5,    0,    3,    0],
       [  77,    7,  943,   13,  418,    1,  297,    0,   25,    0],
       [ 180,   45,   40, 1349,  108,    0,   49,    0,   22,    0],
       [  41,   12,  401,   87, 1018,    0,  242,    0,   13,    0],
       [   9,    0,   10,    2,   31, 1040,    6,  395,  101,  221],
       [ 441,   17,  350,   63,  425,    0,  474,    0,   59,    1],
       [   0,    0,    0,    0,    0,   78,    0, 1454,   11,  196],
       [  49,    3,   28,   22,   39,   21,   64,    5, 1603,    5],
       [   6,    0,    4,    1,   19,   95,    1,  103,    4, 1620]],
      dtype=int16)

>>> train_metrics['acc']
0.692

>>> y_val_hat = lucid.predict(x_train=X_val)
>>> val_metrics = metrics(y_val, y_val_hat, type='classification')
>>> val_metrics['acc']
0.6911

"""

# ----------------------------------------------------------------------- RANDOM GRID SEARCH WITH MEMORY -- # 

# -- Quick tests with less samples
# X_train = X_train[0:1000, :]
# y_train = y_train[0:1000]
# X_val = X_val[0:1000, :]
# y_val = y_val[0:1000]

# grid values
grid_alpha = list(np.arange(0.06, .10, 0.005).round(decimals=4))[1:]

# random shuffle
np.random.shuffle(grid_alpha)

# callback
es_callback = {'earlyStopping': {'metric': 'acc', 'threshold': 0.75}}

# random GridSearch
ds = gridsearch(lucid, X_train, y_train, X_val, y_val, grid_alpha=grid_alpha,
                es_call=es_callback, metric_goal=0.75, fit_epochs=100, grid_iterations=10)
