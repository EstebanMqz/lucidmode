
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: LucidNet                                                                                   -- #
# -- Description: A Lightweight Framework for Transparent and Interpretable FeedForward Neural Net       -- #
# -- models.py: python script with Machine Learning Models                                               -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/LucidNet                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# import other scripts from the project
import data as dt

# import libraries for this script
import numpy as np

# Load data
ohlc_prices = dt.dataset('eth_ohlcv_H8')

# ------------------------------------------------------------------------------------------------------ -- #
# ------------------------------------------------------------------- FEEDFORWARD MULTILAYER PERECEPTRON -- #
# ------------------------------------------------------------------------------------------------------ -- #

class MLP():
    """
    Multi-Layer FeedForward Preceptron:
        1 input layer
        N-hidden layers
        1 output layer
    
    Layer transformations:
        - none
        - convolution
   
    Activation functions:
        For hidden -> Sigmoid, Tanh, ReLu
        For output -> Linear, Sigmoid, Softmax
    
    Weights Initialization:
        - Xavier normal, Xavier uniform, common uniform, according to [1]
        - He, according to [2]
        - Load from object.
    
    Training Schemes:
        - Gradient Descent (Train with all data)
        - Stochastic Gradient Descent (Train with 1 data at a time)
        - Mini-Batch (Train with N data)   
        - Nesterov
        - Adam
    
    Regularization:
        - Types: l1, l2, elasticnet, dropout
        
        - In Cost Function:
            - Weights values of all layers (l1, l2, elasticnet)
        
        - In layers
            - Weights gradient values (l1, l2, elasticnet)
            - Bias gradient values (l1, l2, elasticnet)
            - Neurons activation (dropout)

    Cost Functions: 
        - For classification: 
            - Binary Cross-Entropy 
            - Multiclass Cross-Entropy

        - For regression:
            - Mean Squared Error
    
    Execution Tools:
        - Preprocessing input data: Scale, Standard, Robust Standard.
        - Callback for termination on NaN (cost functions divergence).
        - Callback for early stopping on a metric value difference between Train-Validation sets.
        - Save weights to external object/file.
        - Load weights from external object/file.

    Visualization/Interpretation Tools: 
        - Weight values per layer (Colored bar for each neuron, separation of all layers).
        - CostFunction (train-val) evolution (two lines plot with two y-axis).
        - Convolution operation between layers.
    
    Methods
        - __private: init, init_weights, activation, forward, backward, derivative
        - _protected: train, cost, metrics, regularization, callback
        - public: fit, predict, explain, load, save
    
    Special

    "Ubuntu does not mean that people should not address themselves, the question, therefore is, are you
     going to do so in order to enable the community around you?", Nelson Mandela, 2006. Recorded in a 
     video made previously to the launch of Ubuntu linux distribution.

    - ubuntu_fit: Distributed learning using parallel processing among mini-batches of data selected by its 
                  value on an information divergence matrix.

    - ubuntu_predict: Voting system (classification) or Average system (regression).

    """

    # -------------------------------------------------------------------------------- CLASS CONSTRUCTOR -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def __init__(self, l_hidden, n_hidden, a_hidden, n_output, a_output):
        """
        """
        
        # Number of hidden layers (int)
        self.l_hidden = l_hidden

        # Number of neurons per hidden layer (list of ints, with length n_hidden)
        self.n_hidden = n_hidden

        # Activation of hidden layers (list of str, with length n_hidden)
        self.a_hidden = a_hidden
        
        # Number of neurons in output layer (int)
        self.n_output = n_output

        # Activation of output layer (str)
        self.a_output = a_output
    
    # -------------------------------------------------------------------------- HIDDEN LAYERS FORMATION -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def hidden_layers(self, n_layers, s_layers, a_layers, i_layers, r_layers, s_output, a_output):
        """
        
        n_layers: int
            number of layers

        s_layers: list (of ints, with size of n_layers)
            list with each layer size
        
        a_layers: list (of str, with size of n_layers)
            list with each layer activation
        
        i_layers: list (of str, with size of n_layers)
        
            list with each layer criteria for weights initialization, with options: 

                'xavier_normal': Xavier factor & standard-normally distributed random weights [1]
                'xavier_uniform': Xavier factor & uniformly distributed random weights [1]
                'common-uniform': Commonly used factor & uniformly distributed random weights [1]
                'he': Factor [2]
        
        r_layers: list (of str, with size of n_layers)
            list with each layer regularization criteria, options are:

                'l1': 
                'l2': 
                'elasticnet': 
                'dropout': 

        References
        ----------
        
        [1]: (Glorot and Bengio, 2010)

        [2]: (HE)


        """

        # -- layers container

        # Hidden layers
        self.layers = {'hl_' + layer: {'W': {}, 'b':{}, 'a': {}, 'r':{}} for layer in np.arange(n_layers)}
        # Output layer
        self.layers.update({'ol_' + out: {'a': ''} for out in np.arange(s_output)})

        # iterative layer formation loop
        for layer in np.arange(n_layers):
            # layer neurons composition
            self.layers['hl_' + layer]['W'] = np.zeros((s_layers[layer], 1))
            # layer biases
            self.layers['hl_' + layer]['b'] = np.zeros((s_layers[layer], 1))
            # layer activation
            self.layers['hl_' + layer]['a'] = a_layers[layer]
            # layer regularization
            self.layers['hl_' + layer]['r'] = r_layers[layer]
            # layer weights initialization
            self.layers['hl_' + layer]['i'] = i_layers[layer]
            
            # check that the layer formation was performed OK
            assert(self.layers['hl_' + layer]['W'].shape == (s_layers[layer], s_layers[layer-1]))
            assert(self.layers['hl_' + layer]['b'].shape == (s_layers[layer], s_layers[layer-1]))
        
        # multiplication factor (depends on the activation function) according to [1]
        mf = 6
        # number of inputs (and size of the input layer)
        nx = 4
        # number of outputs (and size of the output layer)
        ny = 1
        # number of neurons of a layer
        nn = 4
        # number of hidden layers
        nh = 2

        # iterative layer formation loop
        for layer in np.arange(n_layers):

            # according to (Glorot and Bengio, 2010)
            if type == 'xavier-uniform':
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(6/(nx + ny))
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.uniform(-r, r, size=(nh, nx)) 
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((ny, 1))

            # to reproduce example results of this notebook
            elif type == 'xavier-standard':
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.randn(nn, nx) * np.sqrt(2/(nx + nh))
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((ny, 1))

            # according to (Glorot and Bengio, 2009)
            elif type == 'common-uniform':
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(1/nx)
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.uniform(-r, r, size=(nn, nx))
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((ny, 1))
            
            elif type == 'zeros':
                # layer neurons composition
                self.layers['hl_' + layer]['W'] = np.zeros((s_layers[layer], 1))
                # layer biases
                self.layers['hl_' + layer]['b'] = np.zeros((s_layers[layer], 1))
            
            # according to (Glorot and Bengio, 2009)
            elif type == 'HE':
                print('pending')

            else: 
                print('Raise Error')
       