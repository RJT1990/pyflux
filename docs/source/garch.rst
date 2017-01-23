GARCH models
==================================

Introduction
----------

Generalized autoregressive conditional heteroskedasticity (GARCH) models aim to model the conditional volatility of a time series. Let :math:`r_{t}` be the dependent variable, for example the returns of a stock in time :math:`t`. We can model this series as:

.. math::
  
  r_{t} = \mu + \sigma_{t}\epsilon_{t}

Here \mu is the expected value of :math:`r_{t}`, :math:`\sigma_{t}` is the standard deviation of :math:`r_{t}` in time :math:`t`, and :math:`\epsilon_{t}` is an error term for time :math:`t`.

GARCH models are motivated by the desire to model :math:`\sigma_{t}` conditional on past information. A primitive model might be a rolling standard deviation - e.g. a 30 day window - or an exponentially weighted standard deviation. A windowed model imposes an arbitrary cutoff which does not seem desirable. An EWMA is slightly more attractive, but how to select the weighting parameter :math:`\lambda` is not immediate.

ARCH/GARCH models are an alterative model which allow for parameters to be estimated in a likelihood-based model. The basic driver of the model is a weighted average of past squared residuals. These lagged squared residuals are known as ARCH terms. Bollerslev (1986) extended the model by including lagged conditional volatility terms, creating GARCH models. Below is the formulation of a GARCH model:

.. math::
  
  y_{t} \sim N\left(\mu,\sigma_{t}^{2}\right)
.. math::

  \sigma_{t}^{2} = \omega + \alpha\epsilon_{t}^{2} + \beta{\sigma_{t-1}^{2}}

We need to impose constraints on this model to ensure the volatility is over 1, in particular :math:`\omega, \alpha, \beta > 0`. If we want to ensure stationarity, we also need to ensure :math:`\alpha + \beta < 1`.

Once we have estimated parameters for the model, we can perform retrospective analysis on volatility, as well as make forecasts for future conditional volatility.

Example
----------

First let us load some financial time series data from Yahoo Finance: 

.. code-block:: python

   import numpy as np
   import pyflux as pf
   import pandas as pd
   from pandas_datareader import DataReader
   from datetime import datetime
   import matplotlib.pyplot as plt
   %matplotlib inline 

   jpm = DataReader('JPM',  'yahoo', datetime(2006,1,1), datetime(2016,3,10))
   returns = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
   returns.index = jpm.index.values[1:jpm.index.values.shape[0]]
   returns.columns = ['JPM Returns']

   plt.figure(figsize=(15,5));
   plt.plot(returns.index,returns);
   plt.ylabel('Returns');
   plt.title('JPM Returns');

.. image:: http://www.pyflux.com/notebooks/GARCH/output_12_12.png

One way to visualize the underlying volatility of the series is to plot the absolute returns :math:`\mid{y}\mid`: 

.. code-block:: python

   plt.figure(figsize=(15,5))
   plt.plot(returns.index, np.abs(returns))
   plt.ylabel('Absolute Returns')
   plt.title('JP Morgan Absolute Returns');

.. image:: http://www.pyflux.com/notebooks/GARCH/output_14_02.png

There appears to be some evidence of volatility clustering over this period. Let’s fit a GARCH(1,1) model using a point mass estimate :math:`z^{MLE}`:

.. code-block:: python
   
   model = pf.GARCH(returns,p=1,q=1)
   x = model.fit()
   x.summary()

   GARCH(1,1)                                                                                                
   ======================================== =================================================
   Dependent Variable: JPM Returns          Method: MLE                                       
   Start Date: 2006-01-05 00:00:00          Log Likelihood: 6594.7911                         
   End Date: 2016-03-10 00:00:00            AIC: -13181.5822                                  
   Number of observations: 2562             BIC: -13158.188                                   
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Vol Constant              0.0                                                              
   q(1)                      0.0933                                                           
   p(1)                      0.9013                                                           
   Returns Constant          0.0009     0.0065     0.1359   0.8919   (-0.0119 | 0.0137)       
   ==========================================================================================

The standard errors are not shown for transformed variables. You can pass through a ``transformed=False`` argument to ``summary`` to obtain this information for untransformed variables.

We can plot the GARCH latent variables with :py:func:`plot_z`: 

.. code-block:: python

   model.plot_z(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GARCH/output_18_12.png

We can plot the fit with :py:func:`plot_fit`: 

.. code-block:: python
   model.plot_fit(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GARCH/output_20_02.png

And plot predictions of future conditional volatility with :py:func:`plot_predict`: 

.. code-block:: python

   model.plot_predict(h=10)

.. image:: http://www.pyflux.com/notebooks/GARCH/plot_predict_garch.png

If we had wanted predictions in DataFrame form, we could have used :py:func:`predict`:. 

We can view how well we predicted using in-sample rolling prediction with :py:func:`plot_predict_is`: 

.. code-block:: python

   model.plot_predict_is(h=50,figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GARCH/plot_predict_is_garch.png

Class Description
----------

.. py:class:: GARCH(data, p, q, target)

   **Generalized Autoregressive Conditional Heteroskedasticity Models (GARCH)**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   p                    int                                The number of autoregressive lags :math:`\sigma^{2}`
   q                    int                                The number of ARCH terms :math:`\epsilon^{2}`
   target               string or int                      Which column of DataFrame/array to use.
   ==================   ===============================    ======================================

   **Attributes**

   .. py:attribute:: latent_variables

      A pf.LatentVariables() object containing information on the model latent variables, 
      prior settings. any fitted values, starting values, and other latent variable 
      information. When a model is fitted, this is where the latent variables are updated/stored. 
      Please see the documentation on Latent Variables for information on attributes within this
      object, as well as methods for accessing the latent variable information. 

   **Methods**

   .. py:method:: adjust_prior(index, prior)

      Adjusts the priors for the model latent variables. The latent variables and their indices
      can be viewed by printing the ``latent_variables`` attribute attached to the model instance.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      index                int                         Index of the latent variable to change
      prior                pf.Family instance          Prior distribution, e.g. ``pf.Normal()``
      ==================   ========================    ======================================

      **Returns**: void - changes the model ``latent_variables`` attribute


   .. py:method:: fit(method, **kwargs)
      
      Estimates latent variables for the model. User chooses an inference option and the
      method returns a results object, as well as updating the model's ``latent_variables`` 
      attribute. 

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      method               str                         Inference option: e.g. 'M-H' or 'MLE'
      ==================   ========================    ======================================

      See Bayesian Inference and Classical Inference sections of the documentation for the 
      full list of inference options. Optional parameters can be entered that are relevant
      to the particular mode of inference chosen.

      **Returns**: pf.Results instance with information for the estimated latent variables

   .. py:method:: plot_fit(**kwargs)
      
      Plots the fit of the model against the data. Optional arguments include *figsize*,
      the dimensions of the figure to plot.

      **Returns** : void - shows a matplotlib plot

   .. py:method:: plot_ppc(T, nsims)

      Plots a histogram for a posterior predictive check with a discrepancy measure of the 
      user's choosing. This method only works if you have fitted using Bayesian inference.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      T                    function                    Discrepancy, e.g. ``np.mean`` or ``np.max``
      nsims                int                         How many simulations for the PPC
      ==================   ========================    ======================================

      **Returns**: void - shows a matplotlib plot

   .. py:method:: plot_predict(h, past_values, intervals, **kwargs)
      
      Plots predictions of the model, along with intervals.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many steps to forecast ahead
      past_values          int                         How many past datapoints to plot
      intervals            boolean                     Whether to plot intervals or not
      ==================   ========================    ======================================

      Optional arguments include *figsize* - the dimensions of the figure to plot. Please note
      that if you use Maximum Likelihood or Variational Inference, the intervals shown will not
      reflect latent variable uncertainty. Only Metropolis-Hastings will give you fully Bayesian
      prediction intervals. Bayesian intervals with variational inference are not shown because
      of the limitation of mean-field inference in not accounting for posterior correlations.
      
      **Returns** : void - shows a matplotlib plot

   .. py:method:: plot_predict_is(h, fit_once, fit_method, **kwargs)
      
      Plots in-sample rolling predictions for the model. This means that the user pretends a
      last subsection of data is out-of-sample, and forecasts after each period and assesses 
      how well they did. The user can choose whether to fit parameters once at the beginning 
      or every time step.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many previous timesteps to use
      fit_once             boolean                     Whether to fit once, or every timestep
      fit_method           str                         Which inference option, e.g. 'MLE'
      ==================   ========================    ======================================

      Optional arguments include *figsize* - the dimensions of the figure to plot. **h** is an int of how many previous steps to simulate performance on. 

      **Returns** : void - shows a matplotlib plot

   .. py:method:: plot_sample(nsims, plot_data=True)

      Plots samples from the posterior predictive density of the model. This method only works
      if you fitted the model using Bayesian inference.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      nsims                int                         How many samples to draw
      plot_data            boolean                     Whether to plot the real data as well
      ==================   ========================    ======================================

      **Returns** : void - shows a matplotlib plot

   .. py:method:: plot_z(indices, figsize)

      Returns a plot of the latent variables and their associated uncertainty. 

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      indices              int or list                 Which latent variable indices to plot
      figsize              tuple                       Size of the matplotlib figure
      ==================   ========================    ======================================

      **Returns** : void - shows a matplotlib plot

   .. py:method:: ppc(T, nsims)

      Returns a p-value for a posterior predictive check. This method only works if you have 
      fitted using Bayesian inference.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      T                    function                    Discrepancy, e.g. ``np.mean`` or ``np.max``
      nsims                int                         How many simulations for the PPC
      ==================   ========================    ======================================

      **Returns**: int - the p-value for the discrepancy test

   .. py:method:: predict(h, intervals=False)
      
      Returns a DataFrame of model predictions.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many steps to forecast ahead
      intervals            boolean                     Whether to return prediction intervals
      ==================   ========================    ======================================

      Please note that if you use Maximum Likelihood or Variational Inference, the intervals shown 
      will not reflect latent variable uncertainty. Only Metropolis-Hastings will give you fully 
      Bayesian prediction intervals. Bayesian intervals with variational inference are not shown 
      because of the limitation of mean-field inference in not accounting for posterior correlations.
      
      **Returns** : pd.DataFrame - the model predictions

   .. py:method:: predict_is(h, fit_once, fit_method)
      
      Returns DataFrame of in-sample rolling predictions for the model.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many previous timesteps to use
      fit_once             boolean                     Whether to fit once, or every timestep
      fit_method           str                         Which inference option, e.g. 'MLE'
      ==================   ========================    ======================================

      **Returns** : pd.DataFrame - the model predictions

   .. py:method:: sample(nsims)

      Returns np.ndarray of draws of the data from the posterior predictive density. This
      method only works if you have fitted the model using Bayesian inference.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      nsims                int                         How many posterior draws to take
      ==================   ========================    ======================================

      **Returns** : np.ndarray - samples from the posterior predictive density.

References
----------

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal
of Econometrics. April, 31:3, pp. 307–27.

Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with
Estimates of the Variance of United Kingdom Inflation. Econometrica.
50(4), 987-1007.
