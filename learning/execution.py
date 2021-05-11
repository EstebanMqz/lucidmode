
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- execution.py: python script with execution tools and functions                                      -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ----------------------------------------------------------------------------------- PREPROCESSING DATA -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Preprocessing input data: Scale, Standard, Robust Standard.

# -------------------------------------------------------------------------- CALLBACK: ReduceLROnPlateau -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Callback for reducing learning rate when reached a plateau of non improving metric

# ----------------------------------------------------------------------------- CALLBACK: TerminateOnNaN -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Callback for termination on NaN (cost functions divergence).

# ------------------------------------------------------------------------------ CALLBACK: EarlyStopping -- #
# --------------------------------------------------------------------------------------------------------- #

def callback_es(self, epoch):
    """
    Callback for early stopping on a metric value difference between Train-Validation sets.
    """

    if self.callbacks is not None and 'acc' in self.metrics:
        if 'earlyStopping' in list(self.callbacks.keys()):
            if 'acc' in self.callbacks['earlyStopping']['metric']:
                epoch_metric = self.history['acc']['train'][epoch]['acc']
                callback_threshold = self.callbacks['earlyStopping']['threshold']
                if epoch_metric >= callback_threshold:
                    return 'triggered'

# ------------------------------------------------------------------------------------ SAVE/LOAD WEIGHTS -- #
# --------------------------------------------------------------------------------------------------------- #


def save_load_model(model):
    """
    # -- Save weights to external object/file.
    # -- Load weights from external object/file.
    # https://pypi.org/project/h5py/

    """

    

    return 'coming soon'


# ------------------------------------------------------------------------------ RECORD AND DISPLAY LOGS -- #
# --------------------------------------------------------------------------------------------------------- #

# -- Record and display messages and logs from execution activity

def record_logger(data, verbosity, source):
    """
    Record and display logs of execution and results
    
    
    Parameters
    ----------

    data: dict
        With data to be logged. It must be specified as the following
        {'ith-metric': [ith-value]}

    source: str ; {'terminal', 'file'}
        Where to store logs
    
    verbosity: {3: execution & results (detailed), 2: execution & results (compact), 1: results, 0: nothing}
        Level of verbosity on the logs    

    Results
    -------
    
    """

    
    return 1

