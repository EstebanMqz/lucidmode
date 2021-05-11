
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- convolution.py: python script with convolution functions                                            -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

# -- Load other scripts

# ----------------------------------------------------------------------------------- CONVOLUTION LAYERS -- #

"""

Work in progress for 1d convolution layer

References
----------

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html

"""

# The option would be to choose

# 1) Outputs of past layer are inputs for conv layer, 
# 2) Define type of kernel: Gaussian noise
# 3) Specify memory of convolution: from 1 to batch_size
# 4) Specify kernel parameters: 1) Distribution Variance
# 5) Scale output of convolution: between 0 to 1
# 6) To apply an activation function after convolution: sigmoid, tanh, relu

# Challenge 1) How to compute the gradient of J with respect to the output of 1d Conv layer?
# Since output of convolution goes through an activation function, forward and backward are according to 
# such activation

np.random.seed(123) 

# Previous layer example
y = np.cumsum(np.random.randn(10))

# Previous layer number of inputs
prev_n = 10

# Memory to convolve
param_m = 1

# -- Kernel Parameters {'k_0': Type of kernel, 'k_1': Variance}
param_k_0 = 'gaussian'
param_k_1 = 1.5

# -- Fast Fourier Convolution Operation
z = signal.windows.gaussian(param_m, param_k_1)
y_convolved = signal.fftconvolve(y, z, mode='same')

# -- Scaling, Dimensions and Shape
y_convolved = y_convolved/np.max(abs(y_convolved))
y = y/np.max(abs(y))

y_convolved = np.matrix(y_convolved).T

plt.plot(y, label='original')
plt.legend()
plt.title('Original TS - Gaussian Random Walk')
 
plt.plot(y_convolved, label='convolved')
plt.legend()
plt.title('Convolved TS')

plt.show()
