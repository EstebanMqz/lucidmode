
=========
Functions
=========

``lucidmode`` requires ...  .

Cost functions
--------------

.. autofunction:: lucidmode.functions.cost


The binary cross-entropy or logloss cost function was utilized for both of the implemented models. 

.. math::
    J(w)=-\frac{1}{m} \sum_{i=1}^{m} \big[ y_i\ log(p_{i}) + (1-y_{i})\ log(1-p_{i}) \big]

where:
    - :math:`m`: Number of samples.
    - :math:`w`: Model weights.
    - :math:`y_{i}`: The *i-th* ground truth (observed) output.
    - :math:`p_{i}`: The *i-th* probabilistically forecasted output. 

Metrics
-------

.. autofunction:: lucidmode.tools.metrics.metrics
