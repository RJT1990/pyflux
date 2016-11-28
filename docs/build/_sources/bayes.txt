Bayesian Inference
==================================

PyFlux supports Bayesian inference for all the model types on offer.

Interface
----------

To view the current priors, you should print the model's latent variable object. For example:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   print(model.z)

This will outline the current prior assumptions for each latent variable, as well as the variational approximate distribution that is assumed (if you are performing variational inference). To adjust priors, simply use the following method on your model object:

.. py:function:: adjust_prior(index, prior)

   Adjusts the priors of the model. **index** can be an int or a list. **prior** is a prior object, such as :py:Class:`Normal`.

Here is example usage for :py:func:`adjust_prior`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.list_priors()
   model.adjust_prior(2, pf.Normal(0,1))


Methods
----------

There are a number of Bayesian inference options using the :py:func:`fit`: method. These can be chosen with the method argument.

**Black-Box Variational Inference**

Performs Black Box Variational Inference. Currently the fixed assumption is mean-field variational inference with normal approximate distributions. The gradient used in this implementation is the score function gradient. By default we use 24 samples for the gradient which is quite intense (other implementations use 2-8 samples). For your application, less samples may be as effective and quicker. One of the limitations of the implementation right now is BBVI here does not support using mini-batches of data. It is not clear yet how mini-batches would work with model types that have an underlying sequence of latent states - if it is shown to be effective, then this option will be included in future.

.. code-block:: python
   :linenos:

   model.fit(method='BBVI', iterations='10000', optimizer='ADAM')

* *batch_size* : (default : 24) number of Monte Carlo samples for the gradient
* *iterations* : (default : 3000) number of iterations to run
* *optimizer* : (default: RMSProp) RMSProp or ADAM (stochastic optimizers)
* *map_start*: (default: True) if True, starts latent variables using a MAP/PML estimate
* *mini_batch*: (default: None) if an int, then will sample mini-batches from the data of the size selected (e.g. 32). This option does not work for some model types.
* *learning_rate*: (default: 0.001) the learning rate for the optimizer
* *record_elbo* : (default: False) if True, will record ELBO during optimization, which can is stored as ``x.elbo_records`` for a ``BBVIResults()`` object ``x``.

**Laplace Approximation**

Performs a Laplace approximation on the posterior.

.. code-block:: python
   :linenos:

   model.fit(method='Laplace')

**Metropolis-Hastings**

Performs Metropolis-Hastings MCMC. Currently uses 'one long chain' which is not ideal, but works okay for most of the models available. This method applies a warm-up period of half the number of simulations and applies thinning by removing every other sample to reduce correlation.

.. code-block:: python
   :linenos:

   model.fit(method='M-H')

* *map_start* : (default: True) whether to initialize starting values and the covariance matrix using MAP estimates and the Inverse Hessian
* *nsims* : number of simulations for the chain

**Penalized Maximum Likelihood**

Provides a Maximum a posteriori (MAP) estimate. This estimate is not completely Bayesian as it is based on a 0/1 loss rather than a squared or absolute loss. It can be considered a form of modal approximation, when taken together with the Inverse Hessian matrix.

.. code-block:: python
   :linenos:

   model.fit(method='PML')

* *preopt_search* : (default : True) if True will use a preoptimization stage to find good starting values (if the model type has no available preoptimization method, this argument will be ignored). Turning this off will speed up optimization at the risk of obtaining an inferior solution.
