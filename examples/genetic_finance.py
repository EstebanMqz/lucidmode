
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
from lucidmode.tools.processing import gridsearch

# -------------------------------------------------------------------------------------- GENETIC FINANCE -- #
# --------------------------------------------------------------------------------------------------------- #

# read file from files folder
p_path = 'lucidmode/datasets/timeseries/genetic-finance/'

X_train = np.array(pd.read_csv(p_path + 'X_train.csv').iloc[:, 1:])
y_train_num = np.array(pd.read_csv(p_path + 'y_train.csv'))
X_val = np.array(pd.read_csv(p_path + 'X_val.csv').iloc[:, 1:])
y_val_num = np.array(pd.read_csv(p_path + 'y_val.csv'))

train_ohlc = X_train
y_train = np.zeros((y_train_num.shape[0], 1)).astype(int)
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
lucid = NeuralNet(hidden_l=[90, 90],
                  hidden_a=['sigmoid', 'sigmoid'],
                  hidden_r=[{'type': 'l1', 'lmbda': 0.001, 'ratio':0.95},
                            {'type': 'l2', 'lmbda': 0.001, 'ratio':0.95}],
                
                  output_r={'type': 'l2', 'lmbda': 0.001, 'ratio':0.1},
                  output_n=1, output_a='sigmoid')

# Model and implementation case Formation
lucid.formation(cost={'function': 'binary-logloss',
                      'reg': {'type': 'elasticnet', 'lmbda': 0.001, 'ratio':0.95}},
                init={'input_shape': X_train.shape[1], 'init_layers': 'xavier-uniform'},
                optimizer={'type': 'SGD', 'params': {'learning_rate': 0.001, 'batch_size': 0}},
                metrics=['acc'])

# Inspect object contents  (Weights initialization)
# inspect(lucid)

# --------------------------------------------------------------------------------------------------------- #

# cost evolution
lucid.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, epochs=1000, verbosity=3)

# acces to the train history information
history = lucid.history

# Predict train
y_hat = lucid.predict(x_train=X_train).astype(int)
train_metrics = metrics(y_train, y_hat, type='classification')

# Confusion matrix
train_metrics['cm']

# Overall accuracy
train_metrics['acc']

# Predict train
y_val_hat = lucid.predict(x_train=X_val).astype(int)
val_metrics = metrics(y_val, y_val_hat, type='classification')

# Overall accuracy
val_metrics['acc']

# --------------------------------------------------------------------------------------------------------- #

# grid values
grid_alpha = list(np.arange(1e-4, 1e-2, 1e-4).round(decimals=6))[1:]

# random shuffle
np.random.shuffle(grid_alpha)

# callback
es_callback = {'earlyStopping': {'metric': 'acc', 'threshold': 0.70}}

# random GridSearch
ds = gridsearch(lucid, X_train, y_train, X_val, y_val, grid_alpha=grid_alpha,
                es_call=es_callback, metric_goal=0.70, fit_epochs=500, grid_iterations=50)

# --------------------------------------------------------------------------------------------------------- #
# -- OHLC Class plot work in progress

# Plot_3 Observed Class vs Predicted Class
theme_plot_4 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 500},
                    p_labels={'title': 'Clasifications',
                              'x_title': 'Dates', 'y_title': 'Continuous Future Price USD/MXN'})
