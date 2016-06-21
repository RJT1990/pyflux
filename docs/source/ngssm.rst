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

The non-linear local level model (**NLLEV**) model has the options: **NLLEV.Exponential**, **NLLEV.Laplace**, **NLLEV.Poisson**, **NLLEV.t**, 

.. py:class:: NLLEV(data,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

The non-linear local linear trend model (**NLLT**) model has the options: **NLLT.Exponential**, **NLLT.Laplace**, **NLLT.Poisson**, **NLLT.t**, 

.. py:class:: NLLT(data,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

The non-linear dynamic regression model (**NDynLin**) model has the options: **NDynLin.Exponential**, **NDynLin.Laplace**, **NDynLin.Poisson**, **NDynLin.t**, 

.. py:class:: NDynLin(formula,data)

   .. py:attribute:: formula

      patsy notation string describing the regression

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data


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

.. py:function:: fit(method,iterations,step,**kwargs)
   
   Estimates parameters for the model using BBVI. Returns a Results object. **iterations** is the number of iterations for BBVI, and **step** is the step size for RMSProp (default : 0.001).

   Optional arguments include **animate** for the local level and local linear trend models: outputs an animation of stochastic optimization.

Here is example usage for :py:func:`fit`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.fit("M-H",nsims=20000)

.. py:function:: plot_fit(intervals,**kwargs)
   
   Graphs the fit of the model. **intervals** is a boolean; if true shows 95% C.I. intervals for the states.

   Optional arguments include **figsize** - the dimensions of the figure to plot - and **series_type** which has two options: *Filtered* or *Smoothed*.

.. py:function:: plot_parameters(indices, figsize)

   Returns a plot of the parameters and their associated uncertainty. **indices** is a list referring to the parameter indices that you want ot plot. Figsize specifies how big the plot will be.

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

.. py:function:: simulation_smoother(data,beta,H,mu)
   
   Outputs a simulated state trajectory from a simulation smoother. Arguments are **data** : the data to simulate from - use self.data usually - and **beta** : the parameters to use, **H** is the measurement covariance matrix from an approximate Gaussian model, and **mu** is a measurement density constant from an approximate Gaussian model.