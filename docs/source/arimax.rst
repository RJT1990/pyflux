ARIMAX models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from pandas_datareader.data import DataReader
   from datetime import datetime
   import pyflux as pf

   accident_data = # some made-up data (needs to be a DataFrame)

   model = pf.ARIMAX(data=my_data, formula='CarAccidents ~ 1 + Friday', ar=1, ma=1)

Class Arguments
----------

.. py:class:: ARIMAX(data, formula, ar, ma, integ)

   .. py:attribute:: data

      pd.DataFrame : the time-series data

   .. py:attribute:: formula

      patsy notation string describing the regression

   .. py:attribute:: ar

      int : the number of autoregressive lags

   .. py:attribute:: ma

      int : the number of moving average lags

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

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
   model.adjust_prior(2, pf.Normal(0,1))

.. py:function:: fit(method, **kwargs)
   
   Estimates latent variables for the model. Returns a Results object. **method** is an inference/estimation option; see Bayesian Inference and Classical Inference sections for a list of options. If no **method** is provided then a default will be used.

   Optional arguments are specific to the **method** you choose, see the documentation on these methods for more detail.

Here is example usage for :py:func:`fit`:

.. code-block:: python
   :linenos:

   import pyflux as pf

   # model = ... (specify a model)
   model.fit("M-H", nsims=20000)

.. py:function:: plot_fit(**kwargs)
   
   Graphs the fit of the model.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict(h, past_values, intervals, oos_data, **kwargs)
   
   Plots predictions of the model. **h** is an int of how many steps ahead to predict. **past_values** is an int of how many past values of the series to plot. **intervals** is a boolean on whether to include confidence/credibility intervals or not. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_predict_is(h, fit_once, **kwargs)
   
   Plots in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).

   Optional arguments include **figsize** - the dimensions of the figure to plot.

.. py:function:: plot_z(indices, figsize)

   Returns a plot of the latent variables and their associated uncertainty. **indices** is a list referring to the latent variable indices that you want to plot. Figsize specifies how big the plot will be.

.. py:function:: predict(h, oos_data)
   
   Returns DataFrame of model predictions. **h** is an int of how many steps ahead to predict. **oos_data** is a DataFrame in the same format as the original DataFrame and has data for the explanatory variables to be used for prediction.

.. py:function:: predict_is(h, fit_once)
   
   Returns DataFrame of in-sample rolling predictions for the model. **h** is an int of how many previous steps to simulate performance on. **fit_once** is a boolean specifying whether to fit the model once at the beginning of the period (True), or whether to fit after every step (False).