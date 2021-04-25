
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
lucid = Sequential(l_hidden=2, n_hidden=[4, 3], a_hidden=['sigmoid', 'relu'],
                   n_output=1, a_output='sigmoid')

# load example data
data = datasets('xor')
# X train
X = data['x']
# y train
y = data['y']

# initialize weights
lucid.init_weights(n_features=X.shape[1], n_outputs=1, i_layers=['xavier-standard', 'xavier-uniform'])

# Inspect object contents
inspect(lucid)

# fit

# describe

# predict

# save model

# load model
