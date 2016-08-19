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

Performs Black Box Variational Inference. Currently the fixed assumption is mean-field variational inference with normal approximate distributions.

.. code-block:: python
   :linenos:

   model.fit(method='BBVI', iterations='10000', optimizer='ADAM')

* *iterations* : (default : 3000) number of iterations to run
* *optimizer* : (default: RMSProp) RMSProp or ADAM (stochastic optimizers)

**Laplace Approximation**

Performs a Laplace approximation on the posterior.

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

Provides a Maximum a posteriori (MAP) estimate. This estimate is not completely Bayesian as it is based on a 0/1 loss rather than a squared or absolute loss. It can be considered a form of moadl approximation, when taken together with the Inverse Hessian matrix.

.. code-block:: python
   :linenos:

   model.fit(method='PML')


Priors
----------

Priors are contained as classes in the inference module. The following priors are supported:

.. py:class:: InverseGamma(alpha, beta, transform)

   .. py:attribute:: alpha

      the shape parameter for the prior

   .. py:attribute:: beta

      the scale parameter for the prior

   .. py:attribute:: transform

      (default: None) one of ['exp','logit',tanh'] - changes the support of the latent variable.


.. py:class:: Normal(mu0, sigma0, transform)

   .. py:attribute:: mu0

      the location parameter for the prior

   .. py:attribute:: sigma0

      the scale parameter for the prior

   .. py:attribute:: transform

      (default: None) one of ['exp', 'logit', 'tanh'] - changes the support of the latent variable.


.. py:class:: Uniform(transform)

   .. py:attribute:: transform

      (default: None) one of ['exp', 'logit, 'tanh'] - changes the support of the latent variable.

*transform* has implications beyond the prior. For example, if you set an AR(1) prior to a 'tanh' transformation, then the tanh transformation will also carry across to the likelihood, so the parameter that is optimized/estimated is tanh(x) instead of x. This therefore affects models that use Maximum Likelihood (although the prior parameters themselves won't affect the Maximum Likelihood estimate).
