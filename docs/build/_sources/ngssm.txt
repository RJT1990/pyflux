Non-Gaussian State Space models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pyflux as pf
   import pandas as pd

   leicester = pd.read_csv('http://www.pyflux.com/notebooks/leicester_goals_scored.csv')
   leicester.columns= ["Time","Leicester Goals Scored"]

   model = pf.NLLEV.Poisson(data=leicester,target='Leicester Goals Scored')

   fb = DataReader('FB',  'yahoo', datetime(2015,5,1), datetime(2016,5,10))
   returns = pd.DataFrame(np.diff(np.log(fb['Open'].values)))
   returns.index = fb.index.values[1:fb.index.values.shape[0]]
   returns.columns = ['Facebook Returns']

   model2 = pf.NLLEV.t(data=returns,target='Close')


Class Arguments
----------

The **NLLEV()** model (local level) has two factory options:

* *NLLEV.t* : creates a t-distributed model
* *NLLEV.Poisson* : creates a Poisson distributed model

In turn, these creation options have the following arguments:

* *data* : requires a pd.DataFrame object or an np.array
* *integ* : how many times to difference the series (0 = none)
* *target* : (default: None) specify the pandas column name or numpy index if the input is a matrix. If None, the first column will be chosen as the data.

Class Attributes
----------

The **NLLEV()** model object hold the following attributes:

Model Attributes:

* *param_no* : number of model parameters
* *data* : the dependent variable held as a np.array
* *data_name* : string variable containing name of the time series
* *data_type* : whether original datatype is numpy or pandas

Parameter Attributes:

The attribute *param.desc* is a dictionary holding information about individual parameters:

* *name* : name of the parameter
* *index* : index of the parameter (begins with 0)
* *prior* : the prior specification for the parameter
* *q* : the variational distribution approximation

Inference Attributes:

* *params* : holds any estimated parameters
* *ses* : holds any estimated standard errors for parameters (MLE/MAP)
* *ihessian* : holds any estimated inverse Hessian (MLE/MAP)
* *chains* : holds trace information for MCMC runs
* *supported_methods* : which inference methods are supported 
* *default_method* : default inference method
* *self.states* = states
* *self.states_mean* = holds any estimated state means
* *self.states_median* = holds any estimated state medians
* *self.states_upper_95* = holds any estimated state upper 95% credibility inter vals
* *self.states_lower_95* = holds any estimated state lower 95% credibility intervals

Class Methods
----------

**adjust_prior(index,prior)**

Adjusts a prior with the given parameter index. Arguments are:

* *index* : taking a value in range(0,no of parameters)
* *prior* : one of the prior objects listed in the Bayesian Inference section

.. code-block:: python
   :linenos:

   model.list_priors()
   model.adjust_prior(2,ifr.Normal(0,1))

**fit(method)**

Fits parameters for the model. Arguments are:

* *method* : one of ['BBVI',MLE','MAP','M-H','Laplace']
* *nsims* : (default: 100000) how many simulations for M-H
* *smoother_weight* : (default: 0.1) how much weight to give to simulation smoother samples as opposed to the current state

.. code-block:: python
   :linenos:

   model.fit(nsims=20000,smoother_weight=0.01)

**list_priors()**

Lists the current prior specification.

**plot_fit()**

Graphs the fit of the model and the various components.

**plot_predict(h)**

Predicts h timesteps ahead and plots results. Arguments are:

* *h* : (default: 5) how many timesteps to predict ahead
* *past_values* : (default: 20) how many past observations to plot
* *intervals* : (default: True) whether to plot prediction intervals

**plot_predict_is(h)**

Predicts rolling in-sample prediction for h past timestamps and plots results. Arguments are:

* *h* : (default: 5) how many timesteps to predict
* *past_values* : (default: 20) how many past observations to plot
* *intervals* : (default: True) whether to plot prediction intervals

**predict(h)**

Predicts h timesteps ahead and outputs pd.DataFrame. Arguments are:

* *h* : (default: 5) how many timesteps to predict ahead

**predict_is(h)**

Predicts h timesteps ahead and outputs pd.DataFrame. Arguments are:

* *h* : (default: 5) how many timesteps to predict ahead

**simulation_smoother(data, beta)**

Outputs a simulated state trajectory from a simulation smoother. Arguments are:

* *data* : the data to simulate from - use self.data usually.
* *beta* : the parameters to use - use self.params (after fitting a model) usually.
* *H* :  H from an approximate Gaussian model
* *mu* : mu from an approximate Gaussian model

.. code-block:: python
   :linenos:

   model.plot_predict(h=12,past_values=36)