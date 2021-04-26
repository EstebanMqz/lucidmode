
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- propagate.py: python script with forward and backward propagation functions                         -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# -- Load other scripts
import functions as fn

 # --------------------------------------------------------------------------------------------- FORWARD -- #

def forward(self, A, l):
    layer = list(self.layers.keys())[l]
    W = self.layers[layer]['W']
    b = self.layers[layer]['b']

    return np.matmul(A, W.T) + b.T

def forward_activation(self, A_prev, l):
    layer = list(self.layers.keys())[l]

    Z = forward(self, A_prev, l)
    A = fn.sigma(Z, self.layers[layer]['a'])

    return A, Z

def forward_propagate(self, X): 
    
    # memory to store all the values for later use in backward process
    memory = {'A_' + str(i): 0 for i in range(1, len(self.hidden_l) + 3)}
    memory.update({'Z_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    memory.update({'d_' + str(i): 0 for i in range(2, len(self.hidden_l) + 3)})
    memory.update({'dW_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    memory.update({'db_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    Al, memory['A_1'] = X, X

    for l in range(0, len(self.hidden_l) + 1):
        A_prev = Al
        Al, Zl = forward_activation(self, A_prev, l)
        
        # save A and Z for every layer (for backward process)
        memory['Z_' + str(l + 1)] = Zl
        memory['A_' + str(l + 2)] = Al

    return memory

# --------------------------------------------------------------------------------------------- BACKWARD -- #


def backward_propagate(self, memory, Y):
    
    # get the post-activation values for the last layer
    AL = memory['A_' + str(len(self.hidden_l) + 2)]
    Y = Y.reshape(AL.shape)
    m = memory['A_1'].shape[0]
    
    # factor to reproduce results of deep learning coursera and xor data
    wm_i = 100
    
    # first delta for output layer
    dAL = (AL - Y)*fn.d_sigma(memory['Z_' + str(len(self.hidden_l) + 1)], self.output_a)
    memory['d_' + str(len(self.hidden_l) + 2)] = dAL

    # just loop hidden layers since the above was for the outputlayer
    for l in range(len(self.hidden_l) - 1 , -1, -1):

        # layer labels
        layer = list(self.layers.keys())[l]
        layer_1 = list(self.layers.keys())[l+1]

        # dW and db previous layer
        dW = (1/m) * np.dot(memory['d_' + str(l + 3)].T, memory['A_' + str(l + 2)])
        memory['dW_' + str(l + 2)] = dW * wm_i
        db = (1/m) * np.sum(memory['d_' + str(l + 3)]).reshape(dW.shape[0], 1)
        memory['db_' + str(l + 2)] = db * wm_i
        
        # delta of layer
        delta = fn.d_sigma(memory['Z_' + str(l + 1)], self.layers[layer]['a'])
        d = delta * (memory['d_' + str(l + 3)] * self.layers[layer_1]['W'])
        memory['d_' + str(l + 2)] = d

        # check for dimensions
        assert (d.shape == memory['A_' + str(l + 2)].shape)
        assert (dW.shape == self.layers[layer_1]['W'].shape)
        assert (db.shape == self.layers[layer_1]['b'].shape)           

    # last delta for the input layer
    memory['dW_1'] =  (1/m) * np.dot(memory['d_2'].T, memory['A_1']) * wm_i
    memory['db_1'] =  (1/m) * sum(memory['d_2']).reshape(self.layers['hl_0']['W'].shape[0], 1) * wm_i

    return memory
