
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

    def __init__(self, n_hidden, l_hidden, a_hidden, n_output, a_output):
        """
        Class constructor
        
        Parameters
        ----------

        n_hidden: list (of int)
            Number of neurons per hidden layer

        l_hidden: int 
            Number of hidden layers

        a_hidden: list (list of str, with length n_hidden)
            Activation of hidden layers

        n_output: int
            Number of neurons in output layer

        a_output: str
            Activation of output layer (str)

        """
        
        self.n_hidden = n_hidden
        self.l_hidden = l_hidden
        self.a_hidden = a_hidden        
        self.n_output = n_output
        self.a_output = a_output
    
    # -------------------------------------------------------------------------- HIDDEN LAYERS FORMATION -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def layer_formation(self, n_layers, s_layers, a_layers, i_layers, r_layers, s_output, a_output):
        """
        Hidden layers formation with a 0 initialization on weights and bias.
        
        Parameters
        ----------

        n_layers: int
            number of layers

        s_layers: list (of ints, with size of n_layers)
            list with each layer size
        
        a_layers: list (of str, with size of n_layers)
            list with each layer activation
        
        i_layers: list (of str, with size of n_layers)
        
            list with each layer criteria for weights initialization, with options: 

                'common-uniform': Commonly used factor & uniformly distributed random weights [1]
                'xavier_uniform': Xavier factor & uniformly distributed random weights [1]
                'xavier_normal': Xavier factor & standard-normally distributed random weights [1]            
                'he-standard': Factor [2]
        
        r_layers: list (of str, with size of n_layers)
            list with each layer regularization criteria, options are:

                'l1': Lasso regularization |b|
                'l2': Ridge regularization |b|^2
                'elasticnet': C(L1 - L2)
                'dropout': Randomly (uniform) select N neurons in layer and turn its weight to 0
        
        s_output: int
            size of the output layer, or, the number of neurons in the layer
        
        a_output: str
            activation function for output layer

        """

        # Hidden layers
        self.layers = {'hl_' + layer: {'W': {}, 'b':{}, 'a': {}, 'r':{}} for layer in np.arange(n_layers)}

        # Output layer
        self.layers.update({'ol': {'a': a_output, 'W': np.zeros((s_output, 1))}})

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
 

    # --------------------------------------------------------------------------- WEIGHTS INITIALIZATION -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def _init_weights(self, n_features, n_outputs):
        """
        Weight initialization
        
        Parameters
        ----------

        n_features: int
            number of features (inputs) in the model
        
        n_outputs: int
            number of outputs in the model

        References
        ----------
        
        [1] X. Glorot and Y. Bengio.  Understanding the difficulty oftraining deep feedforward neural   
            networks. International Conference on Artificial Intelligence and Statistics, 2010.
        
        [2] He et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet 
            Classification," 2015 IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026-1034, doi: 10.1109/ICCV.2015.123.

        """

        # number of hidden layers, (layers - 1), since the last is always the output
        nh = len(self.layers.keys()) - 1

        # hidden layers weights
        for layer in np.arange(nh):

            # -------------------------------------------------- Number of Neurons in the PREVIOUS layer -- # 
            
            # if is the first layer
            if layer == 0:
                # number of features or model inputs
                nx = n_features
            
            # whether is a hidde or the output layer, is the same.
            else:
                # number of neurons of previous layer
                nx = self.layers['hl_' + layer-1]['W'].shape[0]

            # ------------------------------------------------------ Number of neurons in the NEXT layer -- #
            
            # if is the last hidden layer
            if layer == nh:
                # number of neurons for output layer, or model outputs
                ny = n_outputs
            
            else:
                # number of neurons of following layer
                ny = self.layers['hl_' + layer+1]['W'].shape[0]
            
            # --------------------------------------------------- Number of neurons in the CURRENT layer -- #

            # number of neurons of each layer
            nn = self.layers['hl_' + layer]['W'].shape[0]
            
            # ------------------------------------------ Factor according to activation of CURRENT layer -- #

            # multiplication factor (depends on the activation function) according to [1]
            mf = 4 if self.layers['hl_' + layer]['a'] == 'tanh' else 1

            # As mentioned in [1]
            if type == 'common-uniform':
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(1/nn)
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.uniform(-r, r, size=(ny, nx))
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((nn, 1))

            # According to eq:16 in [1]
            elif type == 'xavier-uniform':
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(6/(nx + ny))
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.uniform(-r, r, size=(ny, nx)) 
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((nn, 1))

            # A variation of the previous, according to [1]
            elif type == 'xavier-standard':
                # Hidden layer weights and biasW
                self.layers['hl_' + layer]['W'] = np.random.randn(ny, nx) * (0 + np.sqrt(2/(nx + ny)))
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((nn, 1))
                       
           # A variation of the previous, according to [1]
            elif type == 'he-standard':
                # Hidden layer weights and bias
                self.layers['hl_' + layer]['W'] = np.random.randn(ny, nx) * (0 + np.sqrt(2/nx))
                # Bias weigths in zero
                self.layers['hl_' + layer]['b'] = np.zeros((nn, 1))

            else: 
                print('Raise Error')
