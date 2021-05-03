
"""""""""
lucidmode
"""""""""

A Lucid Framework for Transparent and Interpretable Machine Learning Models.


.. image:: https://readthedocs.org/projects/lucidmode/badge/?version=main
        :target: https://lucidmode.readthedocs.io/en/main/?badge=main
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/lucidmode/lucidmode
        :target: https://github.com/lucidmode/lucidmode/blob/master/LICENSE
        :alt: License

.. image:: https://img.shields.io/badge/python-v3.8-blue
        :target: https://github.com/lucidmode/lucidmode/
        :alt: Version
        
.. image:: https://img.shields.io/github/commit-activity/m/lucidmode/lucidmode
        :target: https://github.com/lucidmode/lucidmode/
        :alt: Commit history

--------------------
Install dependencies
--------------------

Install all the dependencies stated in the requirements.txt file, just run the following command in terminal:

        pip install -r requirements.txt
        
Or you can manually install one by one using the name and version in the file.

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
- **Activation Functions**: sigmoid, tanh, softmax
- **Cost Functions**: Sum of Squared Error, Binary Cross-Entropy, Multi-Class Cross-Entropy
- **Regularization**: L1, L2, ElasticNet for weights in cost function and in gradient updating
- **Optimization**: Weights optimization with Gradient Descent and learning rate

Complementary
-------------

- **Metrics**: Accuracy, Confusion Matrix (Binary and Multiclass), Confusion Tensor (Multiclass OvR)
- **Visualizations**: Cost evolution
- **Public Datasets**: MNIST, Fashion MNIST
- **Special Datasets**: OHLCV + Symbolic Features of Cryptocurrencies (ETH, BTC)

---------------------------
Author/Principal Maintainer
---------------------------

IFFranciscoME Associate Professor of Financial Engineering and Financial Machine Learning
@ITESO (Western Institute of Technology and Higher Education)

-------
License
-------

**GNU General Public License v3.0** 

*Permissions of this strong copyleft license are conditioned on making available 
complete source code of licensed works and modifications, which include larger 
works using a licensed work, under the same license. Copyright and license notices 
must be preserved. Contributors provide an express grant of patent rights.*

Contact
*For more information in reggards of this repo, please contact if.francisco.me@gmail.com*
