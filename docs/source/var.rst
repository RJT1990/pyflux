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

.. py:class:: VAR(data,ar,ma,integ,target)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

   .. py:attribute:: lags

      int : the order (p) of the VAR

   .. py:attribute:: integ

      int : how many times to difference the time series (default: 0)

   .. py:attribute:: target

      string (data is DataFrame) or int (data is np.array) : which column to use as the time series. If None, the first column will be chosen as the data.

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

.. py:function:: irf(h,shock_index,shock_value,shock_dir,intervals,cumulative)
   
   Plots impulse response function graphs. **h** is how many time steps to look ahead for the effects of the shock, **shock_index** is which variable index to apply the initial shock to, **shock_value** applies a custom shock, but if it is None (default setting) then a 1 standard deviation shock will be applied, **shock_dir** is one of 'positive' or 'negative' and is the direction of the shock, **intervals** specifies whether to plot prediction intervals or not, and **cumulative** is a boolean which specifies whether to plot cumulative effects or not.

.. py:function:: plot_fit(**kwargs)
   
   Graphs the fit of the model.

   Optional arguments include **figsize** - the dimensions of the figure to plot.

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












