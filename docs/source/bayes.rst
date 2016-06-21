Bayesian Inference
==================================

PyFlux supports Bayesian inference for most of its models.

Interface
----------

To view the current priors, you should print the model's parameter object. For example:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   print(model.parameters)

This will outline the current prior assumptions for each parameter, as well as the variational approximate distribution that is assumed (if you are conducting variational inference). To adjust priors, simply use the following method on your model object:

.. py:function:: adjust_prior(index, prior)

   Adjusts the priors of the model. **index** can be an int or a list. **prior** is a prior object, such as :py:Class:`Normal`.

Here is example usage for :py:func:`adjust_prior`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.list_priors()
   model.adjust_prior(2,pf.Normal(0,1))


Methods
----------

There are a number of Bayesian inference options using the :py:func:`fit`: option. These can be chosen with the method option.

**Black-Box Variational Inference**

Performs Black Box Variational Inference. Currently the fixed assumptions are mean-field variational inference with normal approximate distributions.

.. code-block:: python
   :linenos:

   model.fit(method='BBVI',iterations='10000',optimizer='ADAM')

* *iterations* : (default : 30000) number of iterations to run
* *optimizer* : (default: RMSProp) RMSProp or ADAM

**Laplace Approximation**

Performs Laplace Approximation of the posterior.

.. code-block:: python
   :linenos:

   model.fit(method='Laplace')

**Metropolis-Hastings**

Performs Metropolis-Hastings MCMC. Currently uses 'one long chain'.

.. code-block:: python
   :linenos:

   model.fit(method='M-H')

* *simulations* : number of simulations for the chain

**Penalized Maximum Likelihood**

Provides a Maximum a posteriori (MAP) point estimate. This estimate is dubiously Bayesian as it is based on a 0/1 loss rather than a squared or absolute loss. We therefore abide by the naming convention of 'Penalized Maximum Likelihood'.

.. code-block:: python
   :linenos:

   model.fit(method='PML')


Priors
----------

Priors are contained as classes in the the inference module. The following priors are supported:

.. py:class:: InverseGamma(alpha,beta,transform)

   .. py:attribute:: alpha

      the shape parameter for the prior

   .. py:attribute:: beta

      the scale parameter for the prior

   .. py:attribute:: transform

      (default: None) one of ['exp','tanh'] - changes the support of the parameter.


.. py:class:: Normal(mu0,sigma0,transform)

   .. py:attribute:: mu0

      the location parameter for the prior

   .. py:attribute:: sigma0

      the scale parameter for the prior

   .. py:attribute:: transform

      (default: None) one of ['exp','tanh'] - changes the support of the parameter.


.. py:class:: Uniform(transform)

   .. py:attribute:: transform

      (default: None) one of ['exp','tanh'] - changes the support of the parameter.

*transform* has implications beyond the prior. For example, if you set an AR(1) prior to a 'tanh' transformation, then the tanh transformation will also carry across to the likelihood, so the parameter that is optimized/estimated is tanh(x) instead of x.
