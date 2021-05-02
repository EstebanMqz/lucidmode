
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

# -- load datasets
from tools.io_data import datasets

# -- base libraries
import numpy as np

# -- complementary tools
from rich import inspect
from tools.metrics import metrics
from tools.processing import train_val_split

# ------------------------------------------------------------------------------------- IMAGE CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# load example data 
data = datasets('fashion_MNIST')
labels = data['labels']
images = data['images']

# split data
X_train, X_val, y_train, y_val = train_val_split(images, labels, train_size = 0.3, random_state = 1)

# -- Train dataset: X_train.shape(16800, 784) y_train.shape(16800,)
# -- Test dataset: X_train.shape(7200, 784) y_train.test(7200,)

# Neural Net Topology Definition
lucid = NeuralNet(hidden_l=[60, 30, 10], hidden_a=['tanh', 'tanh', 'sigmoid'],
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
inspect(lucid)

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
