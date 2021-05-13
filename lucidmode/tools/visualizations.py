
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models.    -- #
# -- visualizations.py: python script with visualization functions                                       -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
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

# ---------------------------------------------------------------------------------------- IMAGE CATALOG -- #
# --------------------------------------------------------------------------------------------------------- #

# - A matrix of nxm randomly seleceted images for visual exploration

# cols = 10
# rows = 4
# fig, axs = plt.subplots(rows, cols, figsize=(16, 5))
# for i in range(rows):
    #img = 0
    #l = np.nonzero(labels == i)
    #for j in np.random.choice(l[0], cols):
        #axs[i, img].axis('off')
        #hm = images[j, :].reshape(28, 28)
        #axs[i, img].imshow(hm.astype(np.uint8))
        #img += 1
