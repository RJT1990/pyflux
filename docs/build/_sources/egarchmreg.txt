Beta-t-EGARCH in-mean regression models
==================================

Introduction
----------

We can expand the Beta-t-EGARCH in-mean model to include exogenous regressors in both the returns and conditional volatility equation:

.. math::
  
   y_{t} =  \mu + \sum^{m}_{k=1}\phi_{m}{X_{m,t}} + \exp\left(\lambda_{t\mid{t-1}}/2\right)\epsilon_{t}

.. math::

   \lambda_{t\mid{t-1}} = \alpha_{0} + \sum^{p}_{i=1}\alpha_{i}\lambda_{t-i} + \sum^{q}_{j=1}\beta_{j}\left(\frac{\left(\nu+1\right)y_{t-j}^{2}}{\nu\exp\left(\lambda_{t-j\mid{t-j-1}}\right) + y_{t-j}^{2}}-1\right) + \sum^{m}_{k=1}\gamma_{m}{X_{m,t}}

.. math::
  
   \epsilon_{t} \sim t_{\nu}

Developer Note
----------
- This model type has yet to be Cythonized so performance can be slow.

Example
----------

First let us load some financial time series data from Yahoo Finance: 

.. code-block:: python

   import numpy as np
   import pyflux as pf
   import pandas as pd
   from pandas.io.data import DataReader
   from datetime import datetime
   import matplotlib.pyplot as plt
   %matplotlib inline 

   a = DataReader('JPM',  'yahoo', datetime(2000,1,1), datetime(2016,3,10))
   a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
   a_returns.index = a.index.values[1:a.index.values.shape[0]]
   a_returns.columns = ["JPM Returns"]

   spy = DataReader('SPY',  'yahoo', datetime(2000,1,1), datetime(2016,3,10))
   spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
   spy_returns.index = spy.index.values[1:spy.index.values.shape[0]]
   spy_returns.columns = ['S&P500 Returns']

   one_mon = DataReader('DGS1MO', 'fred',datetime(2000,1,1), datetime(2016,3,10))
   one_day = np.log(1+one_mon)/365

   returns = pd.concat([one_day,a_returns,spy_returns],axis=1).dropna()
   excess_m = returns["JPM Returns"].values - returns['DGS1MO'].values
   excess_spy = returns["S&P500 Returns"].values - returns['DGS1MO'].values
   final_returns = pd.DataFrame(np.transpose([excess_m,excess_spy, returns['DGS1MO'].values]))
   final_returns.columns=["JPM","SP500","Rf"]
   final_returns.index = returns.index

   plt.figure(figsize=(15,5))
   plt.title("Excess Returns")
   x = plt.plot(final_returns);
   plt.legend(iter(x), final_returns.columns);

.. image:: http://www.pyflux.com/notebooks/EGARCHMReg/output_6_0.png

Let’s fit an EGARCH-M model to JPM’s excess returns series, with a risk-free rate regressor: 

.. code-block:: python
   
   modelx = pf.EGARCHMReg(p=1,q=1,data=final_returns,formula='JPM ~ Rf')
   results = modelx.fit()
   results.summary()

   EGARCHMReg(1,1)                                                                                           
   ======================================== =================================================
   Dependent Variable: JPM                  Method: MLE                                       
   Start Date: 2001-08-01 00:00:00          Log Likelihood: 9621.2996                         
   End Date: 2016-03-10 00:00:00            AIC: -19226.5993                                  
   Number of observations: 3645             BIC: -19176.9904                                  
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   p(1)                      0.9936                                                           
   q(1)                      0.0935                                                           
   v                         6.5414                                                           
   GARCH-M                   -0.0199    0.023      -0.8657  0.3866   (-0.0649 | 0.0251)       
   Vol Beta 1                -0.0545    0.0007     -77.9261 0.0      (-0.0559 | -0.0531)      
   Vol Beta Rf               -0.0346    1.1161     -0.031   0.9753   (-2.2222 | 2.153)        
   Returns Beta 1            0.0011     0.0011     1.0218   0.3069   (-0.001 | 0.0032)        
   Returns Beta Rf           -1.1324    0.0941     -12.0398 0.0      (-1.3167 | -0.948)       
   ==========================================================================================

Let’s plot the latent variables with :py:func:`plot_z`: 

.. code-block:: python

   modelx.plot_z([5,7],figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/EGARCHMReg/output_10_0.png

For this stock, the risk-free rate has a negative effect on excess returns. For the effects of returns on volatility, we are far more uncertain. We can plot the fit with :py:func:`plot_fit`: 

.. code-block:: python

   modelx.plot_fit(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/EGARCHMReg/output_13_0.png


Class Description
----------

.. py:class:: EGARCHMReg(data, formula, p, q)

   **Long Memory Beta-t-EGARCH Models**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   formula              string                             Patsy notation specifying the regression
   p                    int                                The number of autoregressive lags :math:`\sigma^{2}`
   q                    int                                The number of ARCH terms :math:`\epsilon^{2}`
   ==================   ===============================    ======================================

   **Attributes**

   .. py:attribute:: latent_variables

      A pf.LatentVariables() object containing information on the model latent variables, 
      prior settings. any fitted values, starting values, and other latent variable 
      information. When a model is fitted, this is where the latent variables are updated/stored. 
      Please see the documentation on Latent Variables for information on attributes within this
      object, as well as methods for accessing the latent variable information. 

   **Methods**

   .. py:method:: add_leverage()

      Adds a leverage term to the model, meaning volatility can respond differently to the sign of
      the news; see Harvey and Succarrat (2013). Conditional volatility will now follow:

      .. math::

         \lambda_{t\mid{t-1}} = \alpha_{0} + \sum^{p}_{i=1}\alpha_{i}\lambda_{t-i} + \sum^{q}_{j=1}\beta_{j}u_{t-j} + \kappa\left(\text{sgn}\left(-\epsilon_{t-1}\right)(u_{t-1}+1)\right)

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
      oos_data             pd.DataFrame                Exogenous variables in a frame for h steps
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
      oos_data             pd.DataFrame                Exogenous variables in a frame for h steps
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

Black, F. (1976) Studies of stock price volatility changes. In: Proceedings of the 1976 Meetings
of the American Statistical Association. pp. 171–181.

Fernandez, C., & Steel, M. F. J. (1998a). On Bayesian Modeling of Fat Tails and
Skewness. Journal of the American Statistical Association, 93, 359–371.

Harvey, A.C. & Chakravarty, T. (2008) Beta-t-(E)GARCH. Cambridge Working Papers in Economics 0840,
Faculty of Economics, University of Cambridge, 2008. [p137]

Harvey, A.C. & Sucarrat, G. (2013) EGARCH models with fat tails, skewness and leverage. Computational
Statistics and Data Analysis, Forthcoming, 2013. URL http://dx.doi.org/10.1016/j.csda.2013.09.
022. [p138, 139, 140, 143]

Mandelbrot, B.B. (1963) The variation of certain speculative prices. Journal of
Business, XXXVI (1963). pp. 392–417

Nelson, D. B. (1991) Conditional heteroskedasticity in asset returns: A new
approach. Econometrica 59, 347—370.