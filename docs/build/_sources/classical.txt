Classical Inference
==================================

PyFlux supports Classical inference.

Methods
----------

There are a number of classical inference options using the fit() option. These can be chosen with the method option.

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