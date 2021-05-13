
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- metrics.py: python script with a variaty of useful metrics                                          -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np

# ---------------------------------------------------------------------------------- PERFORMANCE METRICS -- #
# --------------------------------------------------------------------------------------------------------- #

def metrics(y, y_hat, type, use='learning'):
    """
    
    Metrics for classification. for single class One-Vs-One, for multiclass One-Vs-Rest.
    Metrics for regression. R2 and Residuals Tests (autocorrelation, normality, heteroskedasticity)


    Parameters
    ----------

    y: np.array
        Ground truth data

    y_hat: np.array
        Predicted data

    type: str
        The type of model is going to be tested. The options are: 'classification', 'regression'
    
    use: str
        'learning': To measeure performance of models in the learning process
        'information': To measure information aspects for generalization goals

    Results
    -------

    c_metrics: dict
        with the calculated metrics according to type.

        tpr : True Positive Rate also known as sensitivity
        fpr : False Positive Rate (1 - tpr)
        tnr : True Negative Rate also known as specificity
        fnr : False Negative Rate (1 - tnr)
        acc : Accuracy of results (tpr + tnr) / Population
        lr_p : Positive Likelihodd (tpr/fpr)
        lr_n : Negative Likelihodd (fnr/tnr)

    Pending: Precision, Recall, AUC

    """

    # --------------------------------------------------------------------------- CLASSIFICATION METRICS -- # 

    if type == 'classification':

        # Unique classes
        classes = list(np.unique(y))
        
        # Confusion matrix (empty matrix)
        cm = np.zeros((len(classes), len(classes)))

        # Confussion tensor (a matrix of confussion matrices) for One-Vs-Rest approach
        # It is effectively a column vector with shape (n x 1 ), where n is number of classes, so each class
        # has its own confussion matrix and for all classes their confussion matrices are arange in the 
        # column vector.

        # Confusion tensor n x (2 x 2)
        ct = np.zeros((len(classes), 2, 2))

        def _confussion_matrix(y, y_hat, classes):

            tpr = np.round(sum(y[y_hat == classes[0]] == classes[0]), 4)
            fpr = np.round(sum(y[y_hat == classes[1]] == classes[0]), 4)
            tnr = np.round(sum(y[y_hat == classes[1]] == classes[1]), 4)
            fnr = np.round(sum(y[y_hat == classes[0]] == classes[1]), 4)

            # Confussion matrix
            return np.array([[tpr, fpr], [fnr, tnr]]).round(decimals=2)

        # -- SINGLE-CLASS
        if len(classes) == 2:
        
            cm = _confussion_matrix(y, y_hat, classes)
            acc = np.round((cm['tpr'] + cm['tnr'])/(cm['tpr'] + cm['fnr'] + cm['fpr'] + cm['tnr']), 4)

            lr_p = np.round(cm['tpr']/cm['fpr'], 4)
            lr_n = np.round(cm['tnr']/cm['fnr'], 4)

            # Final format
            c_metrics = {'tpr': cm['tpr'], 'fpr': cm['fpr'], 'tnr': cm['tnr'], 'fnr': cm['fnr'],
                         'acc': acc, 'lr_p': lr_p, 'lr_n': lr_n, 'cm': cm}
            
            # Final return
            return c_metrics
        
        # -- MULTI-CLASS 
        else: 
            
            # Global confusion matrix
            for i in range(len(y)):
                cm[y[i]][y_hat[i]] += 1
           
            # Support objects
            c_metrics = {}
            i_ct = 0

            # Metrics for each class using One-Vs-Rest approach
            for nth in classes:

                tpr = np.round(sum(y[y_hat == nth] == nth)/(y.shape[0] - 
                               sum(y[y_hat == nth] == nth)), 4)
                tnr = np.round(sum(y[y_hat == nth] != nth)/(y.shape[0] - 
                               sum(y[y_hat == nth] != nth)), 4)
                fpr = np.round((1 - tnr), 4)
                fnr = np.round((1 - tpr), 4)
                acc = np.round((tpr + tnr)/(tpr + fnr + fpr + tnr), 4)
                lr_p = np.round((tpr/fpr), 4)
                lr_n = np.round((fnr/tnr), 4)

                c_metrics[str(nth)] = {'tpr': tpr, 'tnr': fpr, 'fpr': fpr, 'fnr': fnr,
                                       'acc': acc, 'lr_p': lr_p, 'lr_n': lr_n}

                ct[i_ct] = np.array([[tpr, fpr], [fnr, tnr]]).round(decimals=2)
                i_ct += 1

                # Global accuracy metric
                acc_g = np.round(np.sum(np.diag(cm))/y.shape[0], 4)
           
            # Add confussion matrix to pre-final result
            c_metrics = {'im': c_metrics, 'cm': cm.astype(np.int16), 'ct': ct, 'acc': acc_g}
            
            # Final return
            return c_metrics
    
    # ------------------------------------------------------------------------------- REGRESSION METRICS -- # 

    elif type == 'regression':

        # -- R2
        # -- MSE
        # -- RMSE
        # -- RSS

        return 'coming soon'


# ------------------------------------------------------------------------------------- LEARNING METRICS -- #
# --------------------------------------------------------------------------------------------------------- #

def learning_metrics(data, type):
    """
    Metrics used for the learning process

    
    Parameters
    ----------


    Returns
    ------


    References
    ----------


    """

    # -- Kullback-Liebler

    return 'coming soon'
