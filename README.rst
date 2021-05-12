
|

.. image:: https://raw.githubusercontent.com/lucidmode/lucidmode/main/images/lucidmode_logo.png
        :align: center

|

|

.. image:: https://readthedocs.org/projects/lucidmode/badge/?version=latest
        :target: https://lucidmode.readthedocs.io
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/lucidmode.svg
        :target: https://pypi.python.org/pypi/lucidmode/
        :alt: Version

.. image:: https://img.shields.io/github/license/lucidmode/lucidmode
        :target: https://github.com/lucidmode/lucidmode/blob/master/LICENSE
        :alt: License
        
.. image:: https://img.shields.io/badge/python-v3.8-blue
        :target: https://github.com/lucidmode/lucidmode/
        :alt: Version
        
.. image:: https://badges.pufler.dev/visits/lucidmode/lucidmode
        :target: https://github.com/lucidmode/lucidmode/graphs/traffic
        :alt: Visits     

|

**lucidmode** is an open-source, low-code and lightweight Python framework for transparent and interpretable machine learning models. It has built in machine learning methods optimized for visual interpretation of some of the most relevant calculations.

-------------
Documentation
-------------

- Oficial Website: https://www.lucidmode.org
- Documentation: https://lucidmode.readthedocs.io
- Python Package Index (PyPI) repository: https://pypi.org/project/lucidmode/
- Github repository: https://github.com/lucidmode/lucidmode

------------
Installation
------------

- With package manager *(coming soon)*

Install by using pip package manager:
        
        pip install lucidmode

- Cloning repository
  
Clone entire github project

        git@github.com:lucidmode/lucidmode.git

and then install dependencies

        pip install -r requirements.txt
        
------
Models
------

Artificial Neural Network
-------------------------

Feedforward Multilayer perceptron with backpropagation.

- **fit**: Fit model to data
- **predict**: Prediction according to model

Initialization, Activations, Cost functions, regularization, optimization
-------------------------------------------------------------------------

- **Weights Initialization**: With 4 types of criterias (zeros, xavier, common, he)
- **Activation Functions**: sigmoid, tanh, ReLU
- **Cost Functions**: Sum of Squared Error, Binary Cross-Entropy, Multi-Class Cross-Entropy
- **Regularization**: L1, L2, ElasticNet for weights in cost function and in gradient updating
- **Optimization**: Weights optimization with Gradient Descent (GD, SGD, Batch) with learning rate
- **Execution**: Callback (metric threshold), History (Cost and metrics)
- **Hyperparameter Optimization**: Random Grid Search with Memory

Complementary
-------------

- **Metrics**: Accuracy, Confusion Matrix (Binary and Multiclass), Confusion Tensor (Multiclass OvR)
- **Visualizations**: Cost evolution
- **Public Datasets**: MNIST, Fashion MNIST
- **Special Datasets**: OHLCV + Symbolic Features of Cryptocurrencies (ETH, BTC)

---------------
Important Links
---------------

- Release notes: https://github.com/lucidmode/lucidmode/releases
- Issues: https://github.com/lucidmode/lucidmode/issues
- Example Notebooks: https://github.com/lucidmode/lucidmode/tree/main/notebooks
- Documentation: https://lucidmode.readthedocs.io
- Python Package Index (PyPI) repository: https://pypi.org/project/lucidmode/

---------------------------
Author/Principal Maintainer
---------------------------

Francisco Munnoz (IFFranciscoME) Is an associate professor of financial engineering and financial machine learning ITESO (Western Institute of Technology and Higher Education)

-------
License
-------

**GNU General Public License v3.0** 

*Permissions of this strong copyleft license are conditioned on making available 
complete source code of licensed works and modifications, which include larger 
works using a licensed work, under the same license. Copyright and license notices 
must be preserved. Contributors provide an express grant of patent rights.*

*Contact: For more information in reggards of this repo, please contact francisco.me@iteso.mx*
