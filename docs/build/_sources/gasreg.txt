GAS Regression models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pyflux as pf
   import pandas as pd
   from pandas.io.data import DataReader
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

   model = pf.GASReg('Amazon ~ SP500',data=final_returns, family=pf.GASt()) # dynamic beta model

Class Arguments
----------

.. py:class:: GASReg(formula, data, family)

   .. py:attribute:: formula

      patsy notation string describing the regression

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

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
   
   Graphs the fit of the model and the dynamic betas.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_z(indices, figsize)

   Returns a plot of the latent variables and their associated uncertainty. **indices** is a list referring to the latent variable indices that you want ot plot. Figsize specifies how big the plot will be.

.. py:function:: plot_predict(h,past_values,intervals,oos_data,**kwargs)
   
   Plots predictions of the model. **h** is an int of how many steps ahead to predict. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict_is(h,past_values,intervals,**kwargs)
   
   Plots in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: predict(h, oos_data)
   
   Returns DataFrame of model predictions. **h** is an int of how many steps ahead to predict. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

.. py:function:: predict_is(h)
   
   Returns DataFrame of in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on.