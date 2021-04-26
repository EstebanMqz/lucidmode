
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- visualizations.py: python script with visualization functions                                       -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ------------------------------------------------------------------------------------- WEIGHTS ON LAYER -- #
# --------------------------------------------------------------------------------------------------------- #

# - Weight values per layer (Colored bar for each neuron, separation of all layers).

# ------------------------------------------------------------------------------ COST FUNCTION EVOLUTION -- #
# --------------------------------------------------------------------------------------------------------- #

# - CostFunction (train-val) evolution (two lines plot with two y-axis).

# plot cost evolution
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize=(16, 4))
# plt.plot(list(J.keys()), list(J.values()), color='r', linewidth=3)
# plt.title('Cost over epochs')
# plt.xlabel('epochs')
# plt.ylabel('cost');
# plt.show()

# -------------------------------------------------------------------------------- CONVOLUTION OPERATION -- #
# --------------------------------------------------------------------------------------------------------- #

# - Convolution operation between layers.
