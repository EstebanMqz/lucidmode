.. roadmap:

Roadmap
=======

+-----------------+-----------------------------------------------------------------------------------------+
| Project: lucidmode                                                                                        |
+=================+=========================================================================================+
| Description:    | A Lucid Framework for Interpretable Machine Learning Models                             |
+-----------------+-----------------------------------------------------------------------------------------+
| ROADMAP.rst:    | Notes on planning and progress                                                          |
+-----------------+-----------------------------------------------------------------------------------------+
| Author:         | IFFranciscoME - if.francisco.me@gmail.com                                               |
+-----------------+-----------------------------------------------------------------------------------------+
| License:        | GPL-3.0 License                                                                         |
+-----------------+-----------------------------------------------------------------------------------------+
| Repository:     | https://github.com/lucidmode/lucidmode                                                  |
+-----------------+-----------------------------------------------------------------------------------------+

----
Done
----

Documentation
-------------

- **Addition:** Create ROADMAP.rst with info for the roadmap of the project.
- **Enhacements:** Write README file with srt sintax..

Optimizers
----------

- Modify Gradient Descent to support other two options.
- Add Stochastic Gradient Descent (batch_size == 1).
- Add Batch Gradient Descent (1 < batch_size < n_sample).
- Modification: Rename Sequential class as NeuralNet.
- Documentation: Renamed private methods according to python official docs.

-------
Pending
-------

Fix
---

- **Fix**: ReLU activation for hidden layers.

Additions
---------

- **Addition:** Load/Save weights of NeuralNet object.
- **Addition:** Load/Save topology, weights, history, of NeuralNet object.
- **Addition:** L1, L2, ElasticNet default values.

Enhacements
-----------

- **Enhacements:** Complete all the docstrings in reStructured text of functional and planned methods.
- **Enhacements:** Create Optimizer class with SGD and LM and port SGD from fit method.
- **Enhacements:** Add Levenberg-Marquardt Algorithm for weights optimization.
- **Enhacements:** Add ignore_warnings para in fit for cases of RuntimeWarning in math operations.

Tests
-----

- **Tests:** SGD with different batch_size. 
- **Tests:** SGD with different combination of activation functions in layers. 
- **Tests:** SGD with weight initialization options.
- **Tests:** SGD with sigmoid and Softmax activation functions in output layer.
- **Tests:** SGD with no validation data (only train data).
- **Tests:** SGD with with different configurations and values of L1, L2, ElasticNet.

------------

------------
Global Tasks
------------ 

- Build documentation with ReadTheDocs.
- Add badges to README.rst
- Perform UnitTests.
- Build minimal Landing Page.
- Build first beta version in pypi.org.
- Hire fiverr designer for logo (github).
