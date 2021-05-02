
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lucid Framework for Transparent and Interpretable Machine Learning Models            -- #
# -- optimizers.py: python script with optimization methods                                              -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


# -- Load other scripts
import lucidmode.propagate as prop

# -- Load libraries for script
import numpy as np


# ------------------------------------------------------------------------------------------------------ -- #
# -------------------------------------------------------------------- OPTIMIZATION METHODS FOR TRAINING -- #
# ------------------------------------------------------------------------------------------------------ -- #


class Optimizers:

    # -------------------------------------------------------------------------------- CLASS CONSTRUCTOR -- #
    # -------------------------------------------------------------------------------------------------- -- #

    def __init__(self, learning_rate=0.001, momentum=0.01):
        """
        Optimizer class constructor
        
        Parameters
        ----------

        lr: float
            Learning rate

        momentum: float
            Momentum component, if used, for optimization algorithm

        Return
        ------

        self: Modifications on instance of class Optimizers

        """
        
        # Learning Rate
        self.lr = learning_rate

        # momentum
        self.momentum = momentum
    

    # ---------------------------------------------------------------- Stochastic Gradient Descent (SGD) -- #
    # ----------------------------------------------------------------------------------------------------- #

    
    def SGD(self, batch_size):

        """
        Stochastic Gradient Descent

        - Stochastic Gradient Descent (batch_size == 1)
        - Batch Gradient Descent (1 < batch_size < n_samples)
        - Gradient Descent (batch_size == n_samples)

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        """
        
        return 1


    # ------------------------------------------------------------------------- Levenberg-Marquardt (LM) -- #
    # ----------------------------------------------------------------------------------------------------- #


    def LM():
        """
        Levenberg-Marquardt Algorithm

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        """

        return 1 
