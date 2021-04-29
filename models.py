
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

# -- Load other scripts
import propagate as prop
import functions as fn
import regularization as reg
import metrics as mt

# -- Load libraries for script
import numpy as np

# ------------------------------------------------------------------------------------------------------ -- #
# ------------------------------------------------------------------- FEEDFORWARD MULTILAYER PERECEPTRON -- #
# ------------------------------------------------------------------------------------------------------ -- #

class Sequential:

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
        - convolution
   
    Activation functions:
        For hidden -> Sigmoid, Tanh, ReLu
        For output -> Linear (regression), Sigmoid (binary classification), Softmax (multivariate classification)
    
    Methods
    -------
    
    Weights Initialization:
        - Xavier normal, Xavier uniform, common uniform, according to [1]
        - He, according to [2]
        - Load from object.
    
    Training Schemes:
        - Gradient Descent
            - train: use all the data on each epoch
            - validation: use all the data on each epoch
            - note for FTS: None of particular importance

        - Stochastic Gradient Descent
            - train: use 1 example at a time and iterate through all of them on each epoch
            - validation: use all the data when train finishes, do that on each epoch 
            - note for FTS: Do not shuffle data

        - Mini-Batch
            - train: use a partition or subset of the whole data
            - validation: 

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

    def init_weights(self, input_shape, init_layers):
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
        np.random.seed(2)

        # number of hidden layers
        layers = len(self.hidden_l)

        # hidden layers weights
        for layer in range(0, layers):

            # type of initialization for each layer
            type = init_layers

            # store the type of initialization used for each layer
            self.layers['hl_' + str(layer)]['i'] = type

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
            if type == 'common-uniform':
                
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
            elif type == 'xavier-uniform':
                
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
            elif type == 'xavier-standard':
                
                # Multiplying factor (paper version)
                r = mf * np.sqrt(2/(n_prev + n_next))

                # Multiplying factor (coursera Deeplearning version with XOR data)
                # r = 0.01
                
                # Hidden layer weights and biasW
                self.layers['hl_' + str(layer)]['W'] = np.random.randn(n_next, n_prev) * r
                
                # Output layer
                self.layers['ol']['W'] = np.random.randn(self.output_n, self.hidden_l[-1]) * r
                
                # Bias weigths in zero
                self.layers['hl_' + str(layer)]['b'] = np.zeros((nn, 1))
                self.layers['ol']['b'] = np.zeros((self.output_n, 1))

           # According to [2]
            elif type == 'he-standard':
                
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
        self.init_weights(input_shape=init['input_shape'], init_layers=init['init_layers'])

        # Cost (function and regularization definition)
        self.cost = cost
        
        # Metrics to track progress on learning
        self.metrics = metrics

        # Optimizer
        self.optimizer = optimizer


    # ------------------------------------------------------------------ FIT MODEL PARAMETERS (LEARNING) -- #
    # -------------------------------------------------------------------------------------------------- -- #


    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, alpha=0.1, verbosity=3):
        
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
        
        Returns
        -------

        history: dict
            with dynamic keys and iterated values of selected metrics

        # binary output
        # y_train = data['y'].astype(np.int)

        """ 
               
        # Store evolution of cost and other metrics across epochs
        history = {self.cost['function']: {'train': {}, 'val': {}}}
        history.update({metric: {'train': {}, 'val': {}} for metric in self.metrics})
        
        # ------------------------------------------------------------------------------ TRAINING EPOCHS -- #
        for epoch in range(epochs):
            
            # Forward pass
            memory_train = prop.forward_propagate(self, x_train)
            mem_layer = 'A_' + str(len(self.hidden_l) + 2)

            # If there exists a validation test
            if len(x_val) !=0:

                # Forward pass
                memory_val = prop.forward_propagate(self, x_val)
                y_val_hat = memory_val[mem_layer]
                
                # Cost (validation)
                cost_val = fn.cost(y_val_hat, y_val, self.cost['function'])
                history[self.cost['function']]['val'][epoch] = cost_val

                # value prediction
                y_val_hat = self.predict(x_val)

                # Any other metrics registered to track
                for metric in self.metrics:
                    history[metric]['val'][epoch] = mt.metrics(y_val, y_val_hat, type='classification')
            
            # Probability prediction
            y_train_p = memory_train[mem_layer]

            # Value prediction
            y_train_hat = self.predict(x_train)
            
            # Cost (train)
            cost_train = fn.cost(y_train_p, y_train, self.cost['function'])

            # Regularization components (applied only to train)
            if self.cost['reg']:
                Weights = [self.layers[layer]['W'] for layer in self.layers]
                reg_term = reg.l1_l2_EN(Weights,
                                        type=self.cost['reg']['type'],
                                        lmbda=self.cost['reg']['lmbda'],
                                        ratio=self.cost['reg']['ratio'])/x_train.shape[0]

                # Update current value with regularization term
                cost_train += reg_term

            # Update cost value to history
            history[self.cost['function']]['train'][epoch] = cost_train.astype(np.float32).round(decimals=4)
            
            # Any other metrics registered to track
            for metric in self.metrics:
                history[metric]['train'][epoch] = mt.metrics(y_train, y_train_hat, type='classification')
           
            # Verbosity
            if verbosity == 3:
                print('\n- epoch:', "%3i" % epoch, '\n --------------------------------------- ', 
                      '\n- cost_train:', "%.4f" % history[self.cost['function']]['train'][epoch],
                      '- cost_val:', "%.4f" % history[self.cost['function']]['val'][epoch])
                if self.metrics:
                    for metric in self.metrics:
                        print('- ' + metric + '_train' + ': ' + 
                                "%.4f" % history[metric]['train'][epoch][metric],
                                '- ' + metric + '_val' + ': ' +
                                "%.4f" % history[metric]['val'][epoch][metric])

            # -- Backward pass
            grads = prop.backward_propagate(self, memory_train, y_train)

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
                    regW = reg.l1_l2_EN([W], type=r_t, lmbda=r_l, ratio=r_r)/x_train.shape[0]
                    regb = reg.l1_l2_EN([b], type=r_t, lmbda=r_l, ratio=r_r)/x_train.shape[0]
                
                # No regularization
                else:
                    regW, regb = 0

                # Gradient updating
                self.layers[layer]['W'] = W - (self.optimizer['params']['lr'] * dW) + regW
                self.layers[layer]['b'] = b - (self.optimizer['params']['lr'] * db) + regb
        
        # return cost list
        return history

    # ------------------------------------------------------------------------------- PREDICT WITH MODEL -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def predict(self, x_train):
        """

        PREDICT depends of the activation function and number of outpus in output layer

        """
        
        from propagate import forward_propagate

        # -- SINGLE-CLASS
        if self.output_n == 1:            
            # tested only with sigmoid output
            memory = forward_propagate(self, x_train)
            p = memory['A_' + str(len(self.hidden_l) + 2)]
            thr = 0.5
            indx = p > thr
            p[indx] = 1
            p[~indx] = 0

        # -- MULTI-CLASS 
        else:
            memory = forward_propagate(self, x_train)
            p = np.argmax(memory['A_' + str(len(self.hidden_l) + 2)], axis=1)

        return p
