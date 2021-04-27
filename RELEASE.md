# RELEASES NOTES

## v0.2-beta1.0

Files:
- complete data set: MNIST
- complete data set: 'fashion-MNIST'

Tests passed:
- fashion MNIST
- previous release tests

Topology
- single hidden layer (tested)
- 1 - 2 hidden layers (tested)
- different activation functions among hidden layer

Activation functions:
- For hidden -> Sigmoid, Tanh, ReLU (tested and not working)
- For output -> Softmax

Cost Functions: 
- 'binary-logloss' (Binary-class Cross-Entropy)
- 'multi-logloss' (Multi-class Cross-Entropy)

Metrics:
- Confusion matrix (Multi-class)
- Accuracy (Multi-class)

## v0.1-beta1.0

Tests passed: 
- Random XOR data classification

Sequential model:
- hidden_l: Number of neurons per hidden layer (list of int, with length of l_hidden)
- hidden_a: Activation of hidden layers (list of str, with length l_hidden)   
- output_n: Number of neurons in output layer (1)
- output_a: Activation of output layer (str)

Layer transformations:
- linear

Activation functions:
- For hidden -> Sigmoid, Tanh
- For output -> Sigmoid (Binary)

Weights Initialization:
- Xavier normal, Xavier uniform, common uniform, according to [1]
 
Training Schemes:
- Gradient Descent

Cost Functions: 
- Sum of Squared Error (SSE) or Residual Sum of Squares (RSS)

Metrics:
- Accuracy (Binary)
