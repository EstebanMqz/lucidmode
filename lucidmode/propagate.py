
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- propagate.py: python script with forward and backward propagation functions                         -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# -- Load other scripts
import lucidmode.functions as fn
import lucidmode.tools.metrics as mt
import lucidmode.regularization as rg

# --------------------------------------------------------------------------------- ------------- FORWARD -- #

def __forward(self, A, l):
    layer = list(self.layers.keys())[l]
    W = self.layers[layer]['W']
    b = self.layers[layer]['b']
    f = np.matmul(A, W.T) + b.T
    
    return f.astype(np.float16)

def __forward_activation(self, A_prev, l):
    layer = list(self.layers.keys())[l]
    Z = __forward(self, A_prev, l)
    A = fn._sigma(Z, self.layers[layer]['a'])

    return A, Z

def _forward_propagate(self, X): 
    
    # memory to store all the values for later use in backward process
    memory = {'A_' + str(i): 0 for i in range(1, len(self.hidden_l) + 3)}
    memory.update({'Z_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    memory.update({'d_' + str(i): 0 for i in range(2, len(self.hidden_l) + 3)})
    memory.update({'dW_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    memory.update({'db_' + str(i): 0 for i in range(1, len(self.hidden_l) + 2)})
    Al, memory['A_1'] = X, X

    for l in range(0, len(self.hidden_l) + 1):
        A_prev = Al
        Al, Zl = __forward_activation(self, A_prev, l)
        # save A and Z for every layer (for backward process)
        memory['Z_' + str(l + 1)] = Zl
        memory['A_' + str(l + 2)] = Al

    return memory


# --------------------------------------------------------------------------------------------- BACKWARD -- #


def __backward_propagate(self, memory, Y):
    
    # number of samples to scale gradients values
    m = memory['A_1'].shape[0]
    
    # -- MULTI-CLASS: perform one-hot, simple difference for output delta
    if self.output_n != 1:
        # one-hot encoding
        one_hot = np.zeros(shape=(len(Y), self.output_n))
        one_hot[range(len(Y)), Y] = 1
        Y = one_hot.astype(np.int8)
        one_hot = None
        # get the post-activation values for the last layer
        AL = memory['A_' + str(len(self.hidden_l) + 2)]
        # first delta for output layer
        dAL = (AL - Y)

    # -- SINGLE-CLASS: Plain variable usage, activated difference for output delta
    else: 
        # get the post-activation values for the last layer
        AL = memory['A_' + str(len(self.hidden_l) + 2)] + 1e-20 # to avoid 0 division 
        Y = Y.reshape(AL.shape)   
        # first delta for output layer
        dAL = (AL - Y)*fn._dsigma(memory['Z_' + str(len(self.hidden_l) + 1)], self.output_a)

    # store output layer delta
    memory['d_' + str(len(self.hidden_l) + 2)] = dAL

    # just loop hidden layers since the above was for the outputlayer
    for l in range(len(self.hidden_l) - 1 , -1, -1):

        # layer labels
        layer = list(self.layers.keys())[l]
        layer_1 = list(self.layers.keys())[l+1]

        # dW and db previous layer
        dW = (1/m) * np.dot(memory['d_' + str(l + 3)].T, memory['A_' + str(l + 2)])
        memory['dW_' + str(l + 2)] = dW
        db = (1/m) * np.sum(memory['d_' + str(l + 3)], axis=0).reshape(dW.shape[0], 1)
        memory['db_' + str(l + 2)] = db
        
        # delta of layer
        delta = fn._dsigma(memory['A_' + str(l + 2)], self.layers[layer]['a'])
        d = delta * np.matmul(memory['d_' + str(l + 3)], self.layers[layer_1]['W'])
        memory['d_' + str(l + 2)] = d

        # check for dimensions
        assert (d.shape == memory['A_' + str(l + 2)].shape)
        assert (dW.shape == self.layers[layer_1]['W'].shape)
        assert (db.shape == self.layers[layer_1]['b'].shape)           

    # last delta for the input layer
    memory['dW_1'] =  (1/m) * np.dot(memory['d_2'].T, memory['A_1'])
    memory['db_1'] =  (1/m) * sum(memory['d_2']).reshape(self.layers['hl_0']['W'].shape[0], 1)

    return memory


# --------------------------------------------------------------------------------- FORWARD <-> BACKWARD -- #


def _forward_backward(self, x_train, y_train, x_val=None, y_val=None, epoch=0, verbosity=3, metric_goal=None):
    """
    """

    # -- Calculations for Train set

    # Forward pass
    memory_train = _forward_propagate(self, x_train)
    mem_layer = 'A_' + str(len(self.hidden_l) + 2)

    # Probability prediction and cost value calculation for train
    y_train_p = memory_train[mem_layer]
    y_train_hat = self.predict(x_train)
    cost_train = fn._cost(y_train_p, y_train, self.cost['function'])

    # -- Calculations for Validation set
    if len(x_val) !=0:

        # Forward pass, cost and prediction value calculations for val
        memory_val = _forward_propagate(self, x_val)
        y_val_hat = memory_val[mem_layer]
        cost_val = fn._cost(y_val_hat, y_val, self.cost['function'])
        self.history[self.cost['function']]['val'][epoch] = cost_val
        y_val_hat = self.predict(x_val)
        
        # Other metrics registered to track
        for metric in self.metrics:
            metric_value = mt.metrics(y_val, y_val_hat, type='classification')
            self.history[metric]['val'][epoch] = metric_value
    
    # If there exists regularization in the layer, it is applied only for train.
    if 'reg' in list(self.cost.keys()):
        Weights = [self.layers[layer]['W'] for layer in self.layers]
        cost_train += rg._l1_l2_EN(Weights,
                                   type=self.cost['reg']['type'],
                                   lmbda=self.cost['reg']['lmbda'],
                                   ratio=self.cost['reg']['ratio'])/x_train.shape[0]

    # Update cost value to history
    cost_value = cost_train.astype(np.float32).round(decimals=4)
    self.history[self.cost['function']]['train'][epoch] = cost_value
    
    # Any other metrics registered to track
    for metric in self.metrics:
        metric_value = mt.metrics(y_train, y_train_hat, type='classification')
        self.history[metric]['train'][epoch] = metric_value
    
    # Verbosity
    if verbosity == 3:

        print('\n- epoch:', "%3i" % epoch, '\n --------------------------------------- ', 
              '\n- cost_train:', "%.4f" % self.history[self.cost['function']]['train'][epoch],
              '- cost_val:', "%.4f" % self.history[self.cost['function']]['val'][epoch])

        if self.metrics:
            for metric in self.metrics:

                print('- ' + metric + '_train' + ': ' + 
                        "%.4f" % self.history[metric]['train'][epoch][metric],
                        '- ' + metric + '_val' + ': ' +
                        "%.4f" % self.history[metric]['val'][epoch][metric])                           

    elif verbosity == 2:
        print('coming soon: 2')
    
    elif verbosity == 1:
        print('coming soon: 1')

    # -- Backward pass
    grads = __backward_propagate(self, memory_train, y_train)

    return grads
