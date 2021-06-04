
===============
Release History
===============

v0.4-beta1.0
------------

**Calculation of several metrics for classification**

sensitivity (TPR), specificity (TNR), accuracy (acc), likelihood ratio (positive), likelihood ratio
(negative), confusion matrix (binary and multiclass), confusion tensor (binary for every class in multi-class)

**Sequential Class**

- Move the cost_f and cost_r parameters to be specified from formation method, leave the class instantiation with just the model architecture.
- Move the init_weights method to be specified from formation method.

**Execution**

- Create formation method in the Sequential Class, with the following parameters init, cost, metrics, optimizer.
- Store selected metrics in Train and Validation History

**Visualizations**

- Select metrics for verbose output.

v0.3-beta1.0
------------

**Regularization**

- L1, L2 and ElasticNet on weights and biases, location: gradients
- L1, L2 and ElasticNet on weights and biases, location: cost function

**Numerical Stability**

- in functions.py, in cost, added a 1e-25 value to A, to avoid a divide by zero and invalid multiply cases
  in computations of np.log(A)

**Data Handling**

- train and validation cost

**Visualization**

- print: verbose of cost evolution

**Documentation**

- Improve README

v0.2-beta1.0
------------


**Files**

- complete data set: MNIST
- complete data set: 'fashion-MNIST'

**Tests passed**

- fashion MNIST
- previous release tests

**Topology**

- single hidden layer (tested)
- 1 - 2 hidden layers (tested)
- different activation functions among hidden layer

**Activation functions**

- For hidden -> Sigmoid, Tanh, ReLU (tested and not working)
- For output -> Softmax

**Cost Functions**

- 'binary-logloss' (Binary-class Cross-Entropy)
- 'multi-logloss' (Multi-class Cross-Entropy)

**Metrics**

- Confusion matrix (Multi-class)
- Accuracy (Multi-class)


v0.1-beta1.0
------------

**Tests passed**

- Random XOR data classification

**Sequential model**

- hidden_l: Number of neurons per hidden layer (list of int, with length of l_hidden)
- hidden_a: Activation of hidden layers (list of str, with length l_hidden)   
- output_n: Number of neurons in output layer (1)
- output_a: Activation of output layer (str)

**Layer transformations**

- linear

**Activation functions**

- For hidden -> Sigmoid, Tanh
- For output -> Sigmoid (Binary)

**Weights Initialization**

- Xavier normal, Xavier uniform, common uniform, according to [1]
 
**Training Schemes**

- Gradient Descent

**Cost Functions**

- Sum of Squared Error (SSE) or Residual Sum of Squares (RSS)

**Metrics**

- Accuracy (Binary)
