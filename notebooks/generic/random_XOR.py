
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

# ------------------------------------------------------------------------------------------- RANDOM XOR -- #
# --------------------------------------------------------------------------------------------------------- #

# r in init and wm in fit and no regularization

# load example data XOR
# data = datasets('xor')

# Neural Net Topology Definition
# lucid = NeuralNet(hidden_l=[2], hidden_a=['tanh'], output_n=1, output_a='sigmoid')

# initialize weights
# lucid.init_weights(input_shape=data['x'].shape[1], init_layers=['xavier-standard'])

# Inspect object contents  (Weights initialization)
# inspect(lucid)

# x_train = data['x'].astype(np.float16)
# y_train = data['y'].astype(np.int8)

# cost evolution
# history = lucid.fit(x_train=x_train, y_train=y_train, epochs=1000, alpha=0.1, cost_function='sse')

# Inspect object contents  (Weights final values)
# inspect(lucid)

# predict
# y_hat = lucid.predict(x_train)

# metrics
# acc = accuracy(data['y'], y_hat)
# print(acc)
