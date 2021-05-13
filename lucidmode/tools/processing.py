
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- processing.py: python script with data pre-post processing functions                                -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load other scripts
from lucidmode.tools.metrics import metrics

# -- Load libraries for script
import numpy as np

# ------------------------------------------------------------------------------------- TRAIN_TEST_SPLIT -- #
# --------------------------------------------------------------------------------------------------------- #

def train_val_split(x_data, y_data, train_size=0.8, random_state=1):
    """
    
    To split into train and validation split with an optional third split for final test.

    """

    np.random.seed(random_state)
    arr_rand = np.random.rand(x_data.shape[0])
    split = arr_rand < np.percentile(arr_rand, train_size*100)

    x_train = x_data[split]
    y_train = y_data[split]
    x_val =  x_data[~split]
    y_val = y_data[~split]

    return x_train, x_val, y_train, y_val

# ----------------------------------------------------------------------------------- RANDOM GRID SEARCH -- #
# --------------------------------------------------------------------------------------------------------- #

def gridsearch(model, X_train, y_train, X_val, y_val, metric_goal, fit_epochs, grid_iterations, es_call,
grid_alpha):
    """
    
    params: list of parameters

    random_state: seed for random numbers

    memory: whether to store last value or not

    goal: value of the metric of interest

    """

    # np.random.seed = 123
    n = X_train.shape[0]

    # Early stopping criteria: Number of iterations
    counter = grid_iterations

    # random grid search with memory
    while counter > 0:
        
        # Keep track of grid epochs for early stopping
        counter -= 1
    
        n = X_train.shape[1]             # number of features
        k = len(np.unique(y_train))      # number of classes
        
        grid_model = model

        # -- Parameters to try
        alpha = grid_alpha.pop()
        batch = 0
        
        # -- GRID MODEL FORMATION
        grid_model.formation(cost={'function': 'multi-logloss'},
                             init={'input_shape': X_train.shape[1], 'init_layers': 'xavier-uniform'},
                             optimizer={'type': 'SGD',
                                        'params': {'learning_rate': alpha, 'batch_size': 0}},
                             metrics=['acc'])
        
        # -- GRID MODEL TRAIN LEARNING
        grid_model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, epochs=fit_epochs,
                       verbosity=3, random_state=1, callback=es_call)
        
        # Predict train
        y_hat = grid_model.predict(x_train=X_train)
        train_metrics = metrics(y_train, y_hat, type='classification')

        # Overall accuracy
        acc_train = train_metrics['acc']

        # Predict train
        y_val_hat = grid_model.predict(x_train=X_val)
        val_metrics = metrics(y_val, y_val_hat, type='classification')

        # Overall accuracy
        acc_val = val_metrics['acc']
       
        print('\n-------------------- GRID VALUES -----------------------')
        print(f'alpha={alpha}, lmbda={batch}, acc_train={acc_train}, acc_test={acc_val}')
        print('--------------------------------------------------------\n')
        
        if acc_train > metric_goal:
            metric_goal = acc_train
            
            print('\n\n-------------- A great result was found --------------')
            print('--------------------------------------------------------')
            print(f'alpha={alpha}, lmbda={batch}, acc_train={acc_train}, acc_test={acc_val}')
            print('--------------------------------------------------------')
            print('--------------------------------------------------------\n')

            break
