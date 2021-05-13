
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- models.py: python script with Machine Learning Models                                               -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load other scripts
import lucidmode.propagate as prop
import lucidmode.regularization as reg
import lucidmode.learning.execution as ex

# -- Load libraries for script
import numpy as np

# ------------------------------------------------------------------------------------------------------ -- #
# ------------------------------------------------------------------- FEEDFORWARD MULTILAYER PERECEPTRON -- #
# ------------------------------------------------------------------------------------------------------ -- #


class NeuralNet:

    """
    Artificial Neural Network (Feedforward multilayer pereceptron with backpropagation)

    Topology characteristics
    ------------------------

    hidden_l: Number of neurons per hidden layer (list of int, with length of l_hidden)
    hidden_a: Activation of hidden layers (list of str, with length l_hidden)   
    output_n: Number of neurons in output layer (int)
    output_a: Activation of output layer (str)
   
    Other characteristics
    ---------------------

    Layer transformations:
        - linear
        - (pending) convolution
   
    Activation functions:
        For hidden layers: 
            - Sigmoid, Tanh, ReLu (pending)
        For output layer:
            - Softmax, Linear (pending), Sigmoid (pending)
    
    Methods
    -------
    
    Weights Initialization:
        - Xavier normal, Xavier uniform, common uniform, according to [1]
        - He, according to [2]
        - (pending) Save/Load from object.
    
    Training Schemes:
        - Gradient Descent (Use all data)
            - train: use all the data on each epoch
            - validation: use all the data on each epoch
            - (pending) note for FTS: None of particular importance

        - Stochastic Gradient Descent (use 1 sample)
            - train: use 1 saample at a time and iterate through all of them on each epoch
            - validation: use all the data when train finishes, do that on each epoch 
            - (pending) note for FTS: Do not shuffle data

        - Mini-Batch Gradient Descent (use N samples)
            - train: use a partition or subset of the whole data
            - validation: use a partition or subset of the whole data (same as train)
            - (pending) note for FTS: Do not shuffle data

        - (pending) Adapting Learning
            - Momentum
            - Nesterov
        
        - (pending) Levenberg-Marquardt Algorithm

    Regularization:
        - Types: l1, l2, elasticnet, dropout
        
        - In Cost Function:
            - Weights values of all layers (l1, l2, elasticnet)
        
        - In layers
            - (pending) Weights gradient values (l1, l2, elasticnet)
            - (pending) Bias gradient values (l1, l2, elasticnet)
            - (pending) Neurons activation (dropout)

    Cost Functions: 
        - For classification: 
            - Binary Cross-Entropy 
            - Multiclass Cross-Entropy

        - For regression:
            - Mean Squared Error
    
    Execution Tools:
        - (pending) Preprocessing input data: Scale, Standard, Robust Standard.
        - (pending) Callback for termination on NaN (cost functions divergence).
        - (pending) Callback for early stopping on a metric value difference between Train-Validation sets.
        - (pending) Save weights to external object/file.
        - (pending) Load weights from external object/file.

    Visualization/Interpretation Tools: 
        - (pending) Weight values per layer (Colored bar for each neuron, separation of all layers).
        - (pending) CostFunction (train-val) evolution (two lines plot with two y-axis).
        - (pending) Convolution operation between layers.
    
    General Methods List:
        - public: fit, predict
    
    Special:
    --------

    "Ubuntu does not mean that people should not address themselves, the question, therefore is, are you
     going to do so in order to enable the community around you?", Nelson Mandela, 2006. Recorded in a 
     video made previously to the launch of Ubuntu linux distribution.

    - (pending) ubuntu_fit: Distributed learning using parallel processing among mini-batches of data 
                            selected by its value on an information divergence matrix.

    - (pending) ubuntu_predict: Voting system (classification) or Average system (regression).

    """

    # -------------------------------------------------------------------------------- CLASS CONSTRUCTOR -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def __init__(self, hidden_l, hidden_a, output_n, output_a, cost=None, 
                 hidden_r=None, output_r=None, optimizer=None):

        """
        ANN Class constructor
        
        Parameters
        ----------

        hidden_l: list (of int)
            Number of neurons per hidden layer

        hidden_a: list (list of str, with length hidden_l)
            Activation of hidden layers

        output_n: int
            Number of neurons in output layer

        output_a: str
            Activation of output layer (str)
        
        hidden_r / output_r: list (of str, of size l_hidden)
            list with each pre-layer weights and biases regularization criteria, options are:

                'l1': Lasso regularization |b|
                'l2': Ridge regularization |b|^2
                'elasticnet': C(L1 - L2)
                'dropout': Randomly (uniform) select N neurons in layer and turn its weight to 0
                   
        cost: str
            cost information for model.
            'function': 'binary-logloss', 'multi-logloss', 'mse'
            'reg': {'type': ['l1', 'l2', 'elasticnet'], 'lambda': 0.001, 'ratio': 0.01}

        init: str
            initialization of weights specified from compile method

        Returns
        -------
        
        self: Modifications on instance of class

        """
        
        # Number of neurons per hidden layer
        self.hidden_l = hidden_l

        # Activation of hidden layers
        self.hidden_a = hidden_a

        # Number of neurons in output layer
        self.output_n = output_n

        # Activation of output layer (str)
        self.output_a = output_a

        # Regularization criteria for pre-output-layer weights and biases
        self.output_r = output_r

        # Cost function definition
        self.cost = cost

        # Cost function definition
        self.optimizer = optimizer

        # Regularization criteria for pre-hidden-layer weights and biases
        self.hidden_r = hidden_r
    
    # --------------------------------------------------------------------------- WEIGHTS INITIALIZATION -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def __init_weights(self, input_shape, init_layers, random_state=1):
        """
        Weight initialization
        
        Parameters
        ----------

        input_shape: int
            number of features (inputs) in the model
                
        init_layers: list (of str, with size of n_layers)
        
            list with each layer criteria for weights initialization, with options: 

                'common-uniform': Commonly used factor & uniformly distributed random weights [1]
                'xavier_uniform': Xavier factor & uniformly distributed random weights [1]
                'xavier_normal': Xavier factor & standard-normally distributed random weights [1]            
                'he-standard': Factor [2]
        
        References
        ----------
        
        [1] X. Glorot and Y. Bengio.  Understanding the difficulty oftraining deep feedforward neural   
            networks. International Conference on Artificial Intelligence and Statistics, 2010.
        
        [2] He et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet 
            Classification," 2015 IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026-1034, doi: 10.1109/ICCV.2015.123.

        """

        # reproducibility
        np.random.seed(random_state)

        # number of hidden layers
        layers = len(self.hidden_l)

        # hidden layers weights
        for layer in range(0, layers):

            # store the type of initialization used for each layer
            self.layers['hl_' + str(layer)]['i'] = init_layers

            # number of Neurons in layer
            nn = self.hidden_l[layer]

            # multiplication factor (depends on the activation function) according to [1]
            mf = 4 if self.layers['hl_' + str(layer)]['a'] == 'tanh' else 1

            # check input dimensions for first layer
            if layer == 0:
                n_prev = input_shape
                n_next = self.hidden_l[layer]
            
            # following layers are the same
            else:
                n_prev = self.hidden_l[layer-1]
                n_next = self.hidden_l[layer]

            # As mentioned in [1]
            if init_layers == 'common-uniform':
                
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(1/nn)
                
                # Hidden layer weights and bias
                self.layers['hl_' + str(layer)]['W'] = np.random.uniform(-r, r, size=(n_next, n_prev))
                
                # Output layer
                self.layers['ol']['W'] = np.random.uniform(-r, r, size=(self.output_n, self.hidden_l[-1]))
                
                # Bias weigths in zero
                self.layers['hl_' + str(layer)]['b'] = np.zeros((nn, 1))
                self.layers['ol']['b'] = np.zeros((self.output_n, 1))

            # According to eq:16 in [1]
            elif init_layers == 'xavier-uniform':
                
                # Boundaries according to uniform distribution common heuristic
                r = mf * np.sqrt(6/(n_prev + n_next))
                
                # Hidden layer weights and bias
                self.layers['hl_' + str(layer)]['W'] = np.random.uniform(-r, r, size=(n_next, n_prev))

                # Output layer
                self.layers['ol']['W'] = np.random.uniform(-r, r, size=(self.output_n, self.hidden_l[-1]))
                
                # Bias weigths in zero
                self.layers['hl_' + str(layer)]['b'] = np.zeros((nn, 1))
                self.layers['ol']['b'] = np.zeros((self.output_n, 1))

            # A variation of the previous, according to [1]
            elif init_layers == 'xavier-standard':
                
                # Multiplying factor (paper version)
                r = mf * np.sqrt(2/(n_prev + n_next))
                
                # Hidden layer weights and biasW
                self.layers['hl_' + str(layer)]['W'] = np.random.randn(n_next, n_prev) * r
                
                # Output layer
                self.layers['ol']['W'] = np.random.randn(self.output_n, self.hidden_l[-1]) * r
                
                # Bias weigths in zero
                self.layers['hl_' + str(layer)]['b'] = np.zeros((nn, 1))
                self.layers['ol']['b'] = np.zeros((self.output_n, 1))

           # According to [2]
            elif init_layers == 'he-standard':
                
                # Multiplying factor
                r = mf * np.sqrt(2/(n_prev + n_next))
                
                # Hidden layer weights and bias
                self.layers['hl_' + str(layer)]['W'] = np.random.randn(n_next, n_prev) * r
                
                # Output layer
                self.layers['ol']['W'] = np.random.randn(self.output_n, self.hidden_l[-1]) * r

                # Bias weigths in zero
                self.layers['hl_' + str(layer)]['b'] = np.zeros((nn, 1))
                self.layers['ol']['b'] = np.zeros((self.output_n, 1))

            else: 
                print('Raise Error')

    
    # --------------------------------------------------------------------------------- LAYERS FORMATION -- #
    # -------------------------------------------------------------------------------------------------- -- #


    def formation(self, cost=None, optimizer=None, init=None, metrics=None):
        """
        Neural Network Model Formation.        
        
        Parameters
        ----------
        
        self: Instance of class

        cost: 
            cost_f: Cost function
            cost_r: Cost regularization
        
        optimizer: 
            type: Name of method for optimization
            params: parameters according to method
        
        init:
            weight initialization
        
        metrics: 
            metrics to monitor training


        Returns
        -------
        
        self: Modifications on instance of class

        """

        # Hidden layers
        self.layers = {'hl_' + str(layer): {'W': {}, 'b':{}, 'a': {}, 'r':self.hidden_r[layer]}
                       for layer in range(0, len(self.hidden_l))}

        # Output layer
        self.layers.update({'ol': {'W': {}, 'b': {}, 'a': self.output_a, 'r': self.output_r}})

        # iterative layer formation loop
        for layer in range(0, len(self.hidden_l)):

            # layer neurons composition
            self.layers['hl_' + str(layer)]['W'] = None

            # layer biases
            self.layers['hl_' + str(layer)]['b'] = None

            # layer activation
            # if only 1 activation function was provided, use it for all hidden layers
            act = self.hidden_a[0] if len(self.hidden_a) == 1 else self.hidden_a[layer]
            self.layers['hl_' + str(layer)]['a'] = act

            # layer regularization
            self.layers['hl_' + str(layer)]['r'] = self.hidden_r[layer]

            # layer weights initialization
            self.layers['hl_' + str(layer)]['i'] = ''
        
        # Weights initialization
        self.__init_weights(input_shape=init['input_shape'], init_layers=init['init_layers'])

        # Cost (function and regularization definition)
        self.cost = cost
        
        # Metrics to track progress on learning
        self.metrics = metrics

        # Optimizer
        self.optimizer = optimizer
        
        # Store evolution of cost and other metrics across epochs
        history = {self.cost['function']: {'train': {}, 'val': {}}}
        history.update({metric: {'train': {}, 'val': {}} for metric in self.metrics})
        self.history = history

    # ------------------------------------------------------------------ FIT MODEL PARAMETERS (LEARNING) -- #
    # -------------------------------------------------------------------------------------------------- -- #


    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, alpha=0.1, verbosity=3,
            random_state=1, callback=None, randomize=False):
        
        """
        Train model according to specified parameters

        Parameters
        ----------

        x_train: np.array / pd.Series
            Features data with nxm dimensions, n = observations, m = features
        
        y_train: np.array / pd.Series
            Target variable data, dimensions of: nx1 por binary classification and nxm for multi-class
        
        x_val: np.array / pd.Series
            Same as x_train but with data considered as validation

        y_val: np.array / pd.Series
            Same as y_train but with data considered as validation

        epochs: int
            Epochs to iterate the model training
        
        alpha: float
            Learning rate for Gradient Descent
        
        cost_f: str
            Cost function, options are according to functions.cost

        verbosity: int
            level of verbosity to show progress
            3: cost train and cost val at every epoch
        
        callback: dict
            whether there is a stopping criteria or action

            {'earlyStopping': {'metric': 'acc', 'threshold': 0.80}}
        
        Returns
        -------

        history: dict
            with dynamic keys and iterated values of selected metrics

        # binary output
        # y_train = data['y'].astype(np.int)

        """ 

        # Store callbacks in class
        self.callbacks = callback

        # ------------------------------------------------------------------------------ TRAINING EPOCHS -- #
        
        epoch_count = epochs
        epoch_for = 0

        while epoch_count > 0:

            epoch_count -= 1
            epoch_for += 1

            # reproducibility
            np.random.seed(random_state)
            m_train = x_train.shape[0]

            # -- Stochastic Gradient Descent
            if self.optimizer['type'] == 'SGD':

                # get value for batch size
                batch_size = self.optimizer['params']['batch_size']
                
                # if batch size is 0 then use all of samples
                batch_size = m_train if batch_size == 0 else batch_size

                # No random sampling by default
                s_x_train = x_train
                s_y_train = y_train

                if randomize:
                    # randomize samples
                    perm_train = list(np.random.permutation(m_train))
                    s_x_train = x_train[perm_train, :]
                    s_y_train = y_train[perm_train]

                # number of batches
                n_train = np.trunc(m_train / batch_size).astype(int)

                # iterate over all batches
                for k in range(n_train):
                    
                    batch_x = s_x_train[k*batch_size : (k + 1)*batch_size, :]
                    batch_y = s_y_train[k*batch_size : (k + 1)*batch_size]

                    grads = prop._forward_backward(self, batch_x, batch_y, x_val=x_val, y_val=y_val,
                                                   epoch=epoch_for)

                    # ------------------------------------------------------------------- CALLBACK CHECK -- #
                    if ex.callback_es(self, epoch_for) == 'triggered':
                        return 'callback triggered'
                    
                # If remaining batch is left, iterate over it
                if m_train % batch_size != 0:
                    
                    num_last_batch = m_train - (batch_size * n_train)
                    batch_x = s_x_train[m_train - num_last_batch:m_train, :]
                    batch_y = s_y_train[m_train - num_last_batch:m_train]

                    grads = prop._forward_backward(self, batch_x, batch_y, x_val=x_val, y_val=y_val,
                                                   epoch=epoch_for)

                    # ------------------------------------------------------------------- CALLBACK CHECK -- #
                    if ex.callback_es(self, epoch_for) == 'triggered':
                        return 'callback triggered'

                # Update all layers weights and biases
                for l in range(0, len(self.hidden_l) + 1):

                    # Model Elements
                    layer  = list(self.layers.keys())[l]               
                    dW = grads['dW_' + str(l + 1)]
                    W = self.layers[layer]['W']
                    db = grads['db_' + str(l + 1)]
                    b = self.layers[layer]['b']
                    
                    # If the layer has regularization criteria
                    if self.layers[layer]['r']:
                        r_t = self.layers[layer]['r']['type']
                        r_l = self.layers[layer]['r']['lmbda']
                        r_r = self.layers[layer]['r']['ratio']
                        regW = reg._l1_l2_EN([W], type=r_t, lmbda=r_l, ratio=r_r)/x_train.shape[0]
                        regb = reg._l1_l2_EN([b], type=r_t, lmbda=r_l, ratio=r_r)/x_train.shape[0]
                    
                    # No regularization
                    else:
                        regW, regb = 0

                    # Gradient updating
                    self.layers[layer]['W'] = W - (self.optimizer['params']['learning_rate'] * dW) + regW
                    self.layers[layer]['b'] = b - (self.optimizer['params']['learning_rate'] * db) + regb
                                           
                # ----------------------------------------------------------------------- CALLBACK CHECK -- #
                if ex.callback_es(self, epoch_for) == 'triggered':
                    return 'callback triggered'

            # -- levenberg-Marquardt Algorithm
            elif self.optimizer['type'] == 'LMA':
                
                return 'coming soon'


    # ------------------------------------------------------------------------------- PREDICT WITH MODEL -- #
    # -------------------------------------------------------------------------------------------------- -- #


    def predict(self, x_train):
        """

        PREDICT depends of the activation function and number of outpus in output layer

        """
        
        # -- SINGLE-CLASS
        if self.output_n == 1:            
            # tested only with sigmoid output
            memory = prop._forward_propagate(self, x_train)
            p = memory['A_' + str(len(self.hidden_l) + 2)]
            thr = 0.5
            indx = p > thr
            p[indx] = 1
            p[~indx] = 0

        # -- MULTI-CLASS 
        else:
            memory = prop._forward_propagate(self, x_train)
            p = np.argmax(memory['A_' + str(len(self.hidden_l) + 2)], axis=1)

        return p
