Gaussian State Space models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pyflux as pf
   import pandas as pd

   nile = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv')
   nile.index = pd.to_datetime(nile['time'].values,format='%Y')

   model = pf.LLEV(data=niles,target='Poisson') # local level

   USgrowth = pd.DataFrame(np.log(growthdata['VALUE']))
   USgrowth.index = pd.to_datetime(growthdata['DATE'])
   USgrowth.columns = ['Logged US Real GDP']

   model2 = pf.LLT(data=USgrowth) # local linear trend model

Class Arguments
----------

The local level (**LLEV**) and local linear trend (**LLT**) models are of the following form:

.. py:class:: LLEV(data,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

.. py:class:: LLT(data,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

The dynamic linear regression (**DynLin**) model is of the form:

.. py:class:: DynLin(formula,data)

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

.. py:function:: fit(method,**kwargs)
   
   Estimates parameters for the model. Returns a Results object. **method** is an inference/estimation option; see Bayesian Inference and Classical Inference sections for options. If no **method** is provided then a default will be used.

   Optional arguments are specific to the **method** you choose - see the documentation for these methods for more detail.

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

.. py:function:: simulation_smoother(data,beta)
   
   Outputs a simulated state trajectory from a simulation smoother. Arguments are **data** : the data to simulate from - use self.data usually - and **beta** : the parameters to use.
