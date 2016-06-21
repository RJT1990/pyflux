Classical Inference
==================================

PyFlux supports Classical inference.

Methods
----------

There are a number of classical inference options using the :py:func:`fit`: option. These can be chosen with the method option.

**Maximum Likelihood**

Performs Maximum Likelihood estimation.

.. code-block:: python
   :linenos:

   model.fit(method='MLE')

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
