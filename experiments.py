
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
from models import ANN

# -- complementary tools
from rich import inspect

# ---------------------------------------------------------------------------- OHLC TS BINARY CLASSIFIER -- #
# --------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------- OHLC TS REGRESSION -- #
# --------------------------------------------------------------------------------------------------------- #

# Neural Net Topology Definition
lucid = ANN(l_hidden=2, n_hidden=[4, 3], a_hidden=['sigmoid', 'relu'],
            n_output=1, a_output='sigmoid')

# initialize weights
lucid.init_weights(n_features=5, n_outputs=1, i_layers=['xavier-standard', 'xavier-uniform'])

# Inspect object contents
inspect(lucid)
