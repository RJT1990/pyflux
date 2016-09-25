Beta-t-EGARCH-in-mean regression models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pyflux as pf
   from pandas_datareader.data import DataReader
   from datetime import datetime

   data = # some dataframe with financial time series
   model = pf.EGARCHMReg(formula="returns~x+z", data=data, p=1, q=1)
   # here returns is the return series; x and z are exogenous variables to include

Class Arguments
----------

.. py:class:: EGARCHMReg(data, formula, p, q)

   .. py:attribute:: data

      pd.DataFrame : the time-series data for the regression

   .. py:attribute:: formula

      str: patsy notation specifying the regresion; the LHS variable should be the
      return series which the volatility model will be estimated on; the RHS variabls
      are predictors that will be used in the returns and volatility equations.

   .. py:attribute:: p

      int : the number of GARCH terms

   .. py:attribute:: q

      int : the number of score terms

Class Methods
----------

.. py:function:: add_leverage()

   Adds a leverage term to the model to account for the asymmetric effect of new information.

.. py:function:: adjust_prior(index, prior)

   Adjusts the priors of the model. **index** can be an int or a list. **prior** is a prior object, such as Normal(0,3).

Here is example usage for :py:func:`adjust_prior`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.list_priors()
   model.adjust_prior(2, pf.Normal(0,1))

.. py:function:: fit(method,**kwargs)
   
   Estimates latent variables for the model. Returns a Results object. **method** is an inference/estimation option; see Bayesian Inference and Classical Inference sections for options. If no **method** is provided then a default will be used.

   Optional arguments are specific to the **method** you choose - see the documentation for these methods for more detail.

Here is example usage for :py:func:`fit`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.fit("M-H", nsims=20000)

.. py:function:: plot_fit(**kwargs)
   
   Graphs the fit of the model.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_z(indices, figsize)

   Returns a plot of the latent variables and their associated uncertainty. **indices** is a list referring to the latent variable indices that you want to plot. Figsize specifies how big the plot will be.

.. py:function:: plot_predict(h, past_values, intervals, oos_data, **kwargs)
   
   Plots predictions of the model. **h** is an int of how many steps ahead to predict. **past_values** is an int of how many past values of the series to plot. **intervals** is a bool on whether to include confidence/credibility intervals or not. **oos_data** is a 
   DataFrame with the same structure as the original data, but for the out-of-sample period.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict_is(h, fit_once, **kwargs)
   
   Plots in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: predict(h, oos_data)
   
   Returns DataFrame of model predictions. **h** is an int of how many steps ahead to predict.
   **oos_data** is a DataFrame with the same structure as the original data, but for the out-of-sample period. 

.. py:function:: predict_is(h, fit_once)
   
   Returns DataFrame of in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).

