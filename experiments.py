
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

# -- complementary tools
from rich import inspect

# ---------------------------------------------------------------------------- OHLC TS BINARY CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------- OHLC TS REGRESSION -- #
# --------------------------------------------------------------------------------------------------------- #

# Neural Net Topology Definition
lucid = Sequential(hidden_l=[4, 3], hidden_a=['sigmoid', 'relu'], output_n=1, output_a='sigmoid')

# load example data
data = datasets('xor')
# X train
X_train = data['x']
# y train
y = data['y']

# initialize weights
lucid.init_weights(input_shape=X_train.shape[1], init_layers=['xavier-standard', 'xavier-uniform'])

# Inspect object contents
inspect(lucid)
