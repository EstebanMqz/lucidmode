
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- experiments.py: python script with experiment cases                                                 -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- load class
from models import Sequential

# -- load datasets
from data import datasets

# -- base libraries
import numpy as np
import pandas as pd

# -- complementary tools
from rich import inspect
from metrics import accuracy, confusion_matrix

# ------------------------------------------------------------------------------------- IMAGE CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# load example data 
data = datasets('fashion_MNIST')
labels = data['labels']
images = data['images']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 1)

# -- Train dataset: X_train.shape(16800, 784) y_train.shape(16800,)
# -- Test dataset: X_train.shape(7200, 784) y_train.test(7200,)

# Neural Net Topology Definition
lucid = Sequential(hidden_l=[5, 5], hidden_a=['tanh', 'tanh'], output_n=10, output_a='softmax')

# initialize weights
lucid.init_weights(input_shape=X_train.shape[1], init_layers=['common-uniform'])

# Inspect object contents  (Weights initialization)
inspect(lucid)

# cost evolution
history = lucid.fit(x_train=X_train, y_train=y_train, epochs=50, alpha=0.05,
                    cost_function='multi-logloss')

# predict
y_hat = lucid.predict(x_train=X_train)

# metrics
acc = accuracy(y_train, y_hat)
print(acc)

# confussion matrix
cm = confusion_matrix(y_train, y_hat)
print(cm)

# ------------------------------------------------------------------------------------------- RANDOM XOR -- #
# --------------------------------------------------------------------------------------------------------- #

# r in init and wm in fit and no regularization

# load example data XOR
# data = datasets('xor')

# Neural Net Topology Definition
# lucid = Sequential(hidden_l=[2], hidden_a=['tanh'], output_n=1, output_a='sigmoid')

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

# --------------------------------------------------------------------------------- CRYPTO H8 CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- ETH H8 -- #

# ----------------------------------------------------------------------------------------------- BTC H8 -- #
