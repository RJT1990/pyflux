GASX models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import pyflux as pf
   from pandas_datareader.data import DataReader
   from datetime import datetime

   a = DataReader('AMZN',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
   a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
   a_returns.index = a.index.values[1:a.index.values.shape[0]]
   a_returns.columns = ["Amazon Returns"]

   spy = DataReader('SPY',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
   spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
   spy_returns.index = spy.index.values[1:spy.index.values.shape[0]]
   spy_returns.columns = ['S&P500 Returns']

   one_mon = DataReader('DGS1MO', 'fred',datetime(2012,1,1), datetime(2016,6,1))
   one_day = np.log(1+one_mon)/365

   returns = pd.concat([one_day,a_returns,spy_returns],axis=1).dropna()
   excess_m = returns["Amazon Returns"].values - returns['DGS1MO'].values
   excess_spy = returns["S&P500 Returns"].values - returns['DGS1MO'].values
   final_returns = pd.DataFrame(np.transpose([excess_m,excess_spy, returns['DGS1MO'].values]))
   final_returns.columns=["Amazon","SP500","Risk-free rate"]
   final_returns.index = returns.index

   model2 = pf.GASX(formula="Amazon~SP500",data=final_returns,ar=1,sc=1,family=pf.GASSkewt())
   x = model2.fit()
   x.summary()

Class Arguments
----------


.. py:class:: GASX(data,formula,ar,sc,integ,target,family)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: formula

      patsy notation string describing the regression

   .. py:attribute:: ar

      int : the number of autoregressive terms

   .. py:attribute:: sc

      int : the number of score terms

   .. py:attribute:: integ
      
      int : Specifies how many time to difference the time series.

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

   .. py:attribute:: family

      a GAS family object; choices include GASExponential(), GASLaplace(), GASNormal(), GASPoisson(), GASSkewt(), GASt()

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
   
   Estimates latent variables for the model. Returns a Results object. **method** is an inference/estimation option; see Bayesian Inference and Classical Inference sections for options. If no **method** is provided then a default will be used.

   Optional arguments are specific to the **method** you choose - see the documentation for these methods for more detail.

Here is example usage for :py:func:`fit`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.fit("M-H",nsims=20000)

.. py:function:: plot_fit(**kwargs)
   
   Graphs the fit of the model.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_z(indices, figsize)

   Returns a plot of the latent variables and their associated uncertainty. **indices** is a list referring to the latent variable indices that you want ot plot. Figsize specifies how big the plot will be.

.. py:function:: plot_predict(h,past_values,intervals,oos_data,**kwargs)
   
   Plots predictions of the model. **h** is an int of how many steps ahead to predict. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict_is(h, fit_once, **kwargs)
   
   Plots in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: predict(h, oos_data)
   
   Returns DataFrame of model predictions. **h** is an int of how many steps ahead to predict. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

.. py:function:: predict_is(h, fit_once)
   
   Returns DataFrame of in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).