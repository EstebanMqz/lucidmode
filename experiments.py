
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
from metrics import accuracy

# ------------------------------------------------------------------------------------------- RANDOM XOR -- #
# --------------------------------------------------------------------------------------------------------- #

# Neural Net Topology Definition
lucid = Sequential(hidden_l=[2], hidden_a=['sigmoid'], output_n=1, output_a='sigmoid')

# load example data XOR
data = datasets('xor')

# initialize weights
lucid.init_weights(input_shape=data['x'].shape[1], init_layers=['xavier-standard'])

# Inspect object contents  (Weights initialization)
# inspect(lucid)

# cost evolution
J = lucid.fit(data, 1000, 0.1)

# Inspect object contents  (Weights final values)
# inspect(lucid)

# predict
y_test = lucid.predict(data)

# metrics
acc = accuracy(data['y'], y_test)
print(acc)

# --------------------------------------------------------------------------------- CRYPTO H8 CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- ETH H8 -- #

# ----------------------------------------------------------------------------------------------- BTC H8 -- #
