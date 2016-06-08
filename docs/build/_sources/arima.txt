ARIMA models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from pandas.io.data import DataReader
   from datetime import datetime
   import pyflux as pf

   ibm = DataReader('IBM',  'yahoo', datetime(2000,1,1), datetime(2016,3,10))
   ibm['Logged Open'] = np.log(ibm['Open'].values)

   model = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1,target='Logged Open')

Class Arguments
----------

.. py:class:: ARIMA(data,ar,ma,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: ar

      int : the number of autoregressive lags

   .. py:attribute:: ma

      int : the number of moving average lags

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.


Class Attributes
----------

An **ARIMA()** object holds the following attributes:

Model Attributes:

* *ar* : the number of autoregressive lags
* *ma* : the number of moving average lags
* *integ* : order of integration (0 : no difference, 1 : first difference, ...)
* *index* : the timescale of the time-series
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

* *starting_params* : starting parameters for estimation/inference
* *params* : holds any estimated parameters
* *ses* : holds any estimated standard errors for parameters (MLE/MAP)
* *ihessian* : holds any estimated inverse Hessian (MLE/MAP)
* *chains* : holds trace information for MCMC runs
* *supported_methods* : which inference methods are supported 
* *default_method* : default inference method

Class Methods
----------

.. py:function:: adjust_prior(index, prior)

   Adjusts the priors of the model. **index** can be an int or a list. **prior** is a prior object, such as Normal(0,3).

Here is example usage for :py:func:`adjust_prior`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.list_priors()
   model.adjust_prior(2,pf.Normal(0,1))

.. py:function:: fit(method,**kwargs)
   
   Estimates parameters for the model. Returns a Results object. **method** can be one of ['BBVI',MLE','MAP','M-H','Laplace']. 

   Optional arguments include **nsims** - how many simulations if fitting with M-H, **cov_matrix** - option to provide a covariance matrix if fitting with M-H, **iterations** - how many iterations to run if performing BBVI, and **step** - how big should the step size be for RMSprop (default 0.001).

Here is example usage for :py:func:`fit`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.fit("M-H",nsims=20000)

.. py:function:: plot_fit(**kwargs)
   
   Graphs the fit of the model.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict(h,past_values,intervals,**kwargs)
   
   Plots predictions of the model. **h** is an int of how many steps ahead to predict. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict_is(h,past_values,intervals,**kwargs)
   
   Plots in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: predict(h)
   
   Returns DataFrame of model predictions. **h** is an int of how many steps ahead to predict. 

.. py:function:: predict_is(h)
   
   Returns DataFrame of in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on.