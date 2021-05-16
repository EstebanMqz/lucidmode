
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
from numpy.lib.stride_tricks import as_strided
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
https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html

"""

# The option would be to choose

# 1) Outputs of past layer are inputs for conv layer, 
# 2) Define type of kernel: Gaussian noise
# 3) Specify memory of convolution: from 1 to batch_size
# 4) Specify kernel parameters: 1) Distribution Variance
# 5) Scale output of convolution: between 0 to 1
# 6) Apply a block pool function (max, min, mean, median)
# 6) Apply an activation function after convolution: sigmoid, tanh, relu

# Challenge 1) How to compute the gradient of J with respect to the output of 1d Conv layer?
# Option 1: 
#           Since output of convolution goes through an activation function, forward and backward are 
#           according to such activation

np.random.seed(123) 

# Previous layer example
y = np.cumsum(np.random.randn(10))

# Previous layer number of inputs
prev_n = 10

# Memory to convolve
param_m = 10

# -- 1D CONVOLUTION LAYER -- #

"""
Notes: 
Each one-dimensional time series feature is provided to the model as a separate channel or input.

The model will read each input sequence onto a separate set of filter maps, then use a separate kernel to transform each one, essentially learning features from each convolved time series input variable.

1D Convolution operation depends on the kernel transformation which in times depends on the parameters necessary to perform the internal operation. For time series it is necessary to specify a memory value for such operation, that is, at every time t a fixed amount of T values will be used to perform the f(x) transformation.

"""

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


def block_formation(array, block_shape):

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError('block_shape elements must be strictly positive')

    if block_shape.size != array.ndim:
        raise ValueError('block_shape must have the same length as arr_in.shape')

    arr_shape = np.array(array.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError('block_shape is not compatible with arr_in')

    # -- use as_strided in the array to perform the block formation as needed
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(array.strides * block_shape) + array.strides
    arr_out = as_strided(array, shape=new_shape, strides=new_strides, writeable=False)

    return arr_out

# -- 
def block_pool(array, block_shape, func=np.sum):

    if len(block_shape) != array.ndim:
        raise ValueError('block_size must have the same length as array.shape')

    pad_width = []
    for i in range(len(block_shape)):

        if block_shape[i] < 1:
            raise ValueError('Down-sampling factors must be >= 1')

        if array.shape[i] % block_shape[i] != 0:
            after_width = block_shape[i] - (array.shape[i] % block_shape[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    array = np.pad(array, pad_width=pad_width, mode='constant')

    blocked = block_formation(array, block_shape)

    return func(blocked, axis=tuple(range(array.ndim, blocked.ndim)))


# -- Tests 
plt.plot(y, label='original')
plt.legend()
plt.title('Original TS - Convolved TS')
 
plt.plot(y_convolved, label='convolved')
plt.legend()
plt.title('Convolved TS')

plt.show()

# -- FUNCTIONAL POOLING LAYER -- #

# -- Reference pooling with (max, min, mean, median)
function_pooling = block_pool(array=y_convolved, block_shape=(5,1), func=np.mean)

plt.plot(function_pooling, label='function pooling')
plt.legend()
plt.title('Function Pooling of Convolved TS')

plt.show()
