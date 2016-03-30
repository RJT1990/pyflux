Bayesian Inference
==================================

PyFlux supports Bayesian inference.

Interface
----------

**adjust_prior(index,prior_type)**

Adjusts prior for given parameters.

* *index* : the parameter index; can be entered as a list to change multiple priors
* *prior_type* : one of the prior options (see subsequent section)

.. code-block:: python
   :linenos:

   model.adjust_prior(0,pf.Normal(mu0=0,sigma0=5,transform=tanh))

**list_priors()**

Lists priors for the current model.

.. code-block:: python
   :linenos:

   model.list_priors()

**list_q()**

Lists variational distributions for each parameter

.. code-block:: python
   :linenos:

   model.list_q()

Methods
----------

There are a number of Bayesian inference options using the fit() option. These can be chosen with the method option.

**Black-Box Variational Inference**

Performs Black Box Variational Inference.

.. code-block:: python
   :linenos:

   model.fit(method='BBVI',iterations='10000',step='0.001')

* *iterations* : (default : 30000) number of iterations to run
* *step* : (default : 0.001) stepsize for RMSProp

**Laplace Approximation**

Performs Laplace Approximation of the posterior.

.. code-block:: python
   :linenos:

   model.fit(method='Laplace')

**Maximum a Posteriori**

Performs Maximum a posteriori estimation.

.. code-block:: python
   :linenos:

   model.fit(method='MAP')

**Metropolis-Hastings**

Performs Metropolis-Hastings MCMC.

.. code-block:: python
   :linenos:

   model.fit(method='M-H')

* *simulations* : number of simulations for the chain

Priors
----------

Priors are contained as classes in the the inference module. The following priors are currently supported:

**InverseGamma(alpha,beta,transform)**

An Inverse Gamma prior class, with the following arguments:

* *alpha* : the shape parameter for the prior
* *beta* : the scale parameter for the prior
* *transform* : (default: None) one of ['exp','tanh'] - changes the support of the parameter.

**Normal(mu0,sigma0,transform)**

A Normal prior class, with the following arguments:

* *mu0* : the location parameter for the prior
* *sigma0* : the scale parameter for the prior
* *transform* : (default: None) one of ['exp','tanh'] - changes the support of the parameter.

**Uniform(transform)**

A uninformative uniform prior class, with the following arguments:

* *transform* : (default: None) one of ['exp','tanh'] - changes the support of the parameter.

*transform* has implications beyond the prior. For example, if you set an AR(1) prior to a 'tanh' transformation, then the tanh transformation will also carry across to the likelihood, so the parameter that is optimized/estimated is tanh(x) instead of x.
