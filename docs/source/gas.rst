GAS models
==================================

Example
----------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import pyflux as pf

   d = pd.read_csv("https://vincentarelbundock.github/io/Rdatasets/csv/MASS/drivers.csv")
   accidents = d['drivers'].values

   model = pf.GAS(data=accidents,ar=1,sc=1,dist='Poisson')

Class Arguments
----------


.. py:class:: GAS(data,ar,sc,integ,target,family)

   .. py:attribute:: data

      pd.DataFrame or array-like : the time-series data

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