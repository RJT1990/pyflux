Classical Inference
==================================

PyFlux supports classical methods of inference. These can be considered as point mass approximations to the full posterior.

Methods
----------

There are a number of classical inference options using the :py:func:`fit`: method. These can be chosen with the method option.

**Maximum Likelihood**

Performs Maximum Likelihood estimation.

.. code-block:: python
   :linenos:

   model.fit(method='MLE')

* *preopt_search* : (default : True) if True will use a preoptimization stage to find good starting values (if the model type has no available preoptimization method, this argument will be ignored). Turning this off will speed up optimization at the risk of obtaining an inferior solution.

**Ordinary Least Squares**

Performs Ordinary Least Squares estimation.

.. code-block:: python
   :linenos:

   model.fit(method='OLS')

**Penalized Maximum Likelihood**

From a frequentist perspective, PML can be viewed as a type of regularization on the coefficients.

.. code-block:: python
   :linenos:

   model.fit(method='PML')

* *preopt_search* : (default : True) if True will use a preoptimization stage to find good starting values (if the model type has no available preoptimization method, this argument will be ignored). Turning this off will speed up optimization at the risk of obtaining an inferior solution.