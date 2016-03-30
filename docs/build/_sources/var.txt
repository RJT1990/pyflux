VAR models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pyflux as pf
   from pandas.io.data import DataReader
   from datetime import datetime

   ibm = DataReader(['JPM','GS','BAC','C','WFC','MS'],  'yahoo', datetime(2000,1,1), datetime(2016,3,28))
   opening_prices = np.log(ibm['Open'])
   model = pf.VAR(data=opening_prices,lags=1,integ=1)

Class Arguments
----------

The **VAR()** model class has the following arguments:

* *data* : requires a pd.DataFrame object or an np.array
* *lags* : the order (p) of the VAR
* *integ* : (default : 0) order of integration (0 : no difference, 1 : first difference, ...)
* *target* : (default: None) specify the pandas column name or numpy index if the input is a matrix. If None, the first column will be chosen as the data.

Class Attributes
----------

An **VAR()** object holds the following attributes:

Model Attributes:

* *lags* : the order (p) of the VAR
* *integ* : order of integration (0 : no difference, 1 : first difference, ...)
* *index* : the timescale of the time-series
* *data* : the dependent variables held in an np.array
* *data_name* : string variable containing names of the time series
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
* *printed* : (default: True) whether to print output
* *nsims* : (default: 100000) how many simulations if M-H is chosen
* *cov_matrix* (default: None) covariance matrix for M-H
* *iterations* : (default: 30000) how many iterations if BBVI is chosen
* *step* : (default: 0.001) step size for BBVI

.. code-block:: python
   :linenos:

   model.fit("M-H",nsims=20000)

**irf(T,shock_index,shock_dir)**

Plots impulse response function graphs.

* *T* : (default: 5) how many timesteps to predict ahead
* *shock_index* : which variable to apply the shock to
* *shock_value* : (default: None) apply a custom shock, if None then applies 1 std shock
* *shock_dir* : (default: 'positive') one of ['positive','negative']
* *intervals* : (default: True) whether to plot 95/90 prediction intervals      

**list_priors()**

Lists the current prior specification.

**plot_fit()**

Graphs the fit of the model.

**predict(T)**

Predicts T timesteps ahead. Arguments are:

* *T* : (default: 5) how many timesteps to predict ahead
* *lookback* : (default: 20) how many past observations to plot
* *intervals* : (default: True) whether to plot 95/90 prediction intervals

.. code-block:: python
   :linenos:

   model.predict(T=12,lookback=36)