
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
import pandas as pd

# -- complementary tools
from rich import inspect
from lucidmode.tools.metrics import metrics
from lucidmode.tools.io_data import datasets
from lucidmode.tools.processing import gridsearch

# -------------------------------------------------------------------------------------- GENETIC FINANCE -- #
# --------------------------------------------------------------------------------------------------------- #

# load example data 
data = datasets('genetic_finance')

X_train = data['X_train']
X_train = np.array(X_train)

y_train_num = data['y_train']
y_train_num = np.array(y_train_num)
y_train = np.zeros((y_train_num.shape[0], 1)).astype(int)

X_val = data['X_val']
X_val = np.array(X_val)
y_val_num = data['y_val']
y_val_num = np.array(y_val_num)
y_val = np.zeros((y_val_num.shape[0], 1)).astype(int)


# -- Multiclass formulation -- #

# y_train[y_train_num <= -0.25] = 0
# y_train[(-0.25 < y_train_num) & (y_train_num <= 0)] = 1
# y_train[(0 < y_train_num) & (y_train_num <= 0.25)] = 2
# y_train[y_train_num > 0.25] = 3
# y_train = np.squeeze(y_train)

# y_val[y_val_num <= -0.25] = 0
# y_val[(-0.25 < y_val_num) & (y_val_num <= 0)] = 1
# y_val[(0 < y_val_num) & (y_val_num <= 0.25)] = 2
# y_val[y_val_num > 0.25] = 3
# y_val = np.squeeze(y_val)

# -- Binary formulation -- #

y_train[y_train_num <= 0] = 0
y_train[y_train_num > 0] = 1
y_train = np.squeeze(y_train)

y_val[y_val_num <= 0] = 0
y_val[y_val_num > 0] = 1
y_val = np.squeeze(y_val)

# Neural Net Topology Definition
lucid = NeuralNet(hidden_l=[X_train.shape[0], 100, 100], hidden_a=['tanh', 'sigmoid', 'relu'],
                  hidden_r=[{'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                            {'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                            {'type': 'l2', 'lmbda': 0.01, 'ratio':1}],
                
                  output_r={'type': 'l1', 'lmbda': 0.001, 'ratio':0.1},
                  output_n=1, output_a='sigmoid')

# Model and implementation case Formation
lucid.formation(cost={'function': 'binary-logloss', 'reg': {'type': 'l1', 'lmbda': 0.001, 'ratio':0.1}},
                init={'input_shape': X_train.shape[1], 'init_layers': 'xavier-standard'},
                optimizer={'type': 'SGD', 'params': {'learning_rate': 0.007, 'batch_size': 0}},
                metrics=['acc'])

# Inspect object contents  (Weights initialization)
# inspect(lucid)


# grid values
grid_alpha = list(np.arange(1e-5, 1e-2, 1e-3).round(decimals=4))[1:]

# random shuffle
np.random.shuffle(grid_alpha)

# callback
es_callback = {'earlyStopping': {'metric': 'acc', 'threshold': 0.75}}

# random GridSearch
ds = gridsearch(lucid, X_train, y_train, X_val, y_val, grid_alpha=grid_alpha,
                es_call=es_callback, metric_goal=0.75, fit_epochs=100, grid_iterations=10)


# cost evolution
# lucid.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, epochs=200, verbosity=3)

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
