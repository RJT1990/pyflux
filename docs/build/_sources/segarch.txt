Beta Skew-t GARCH models
==================================

Introduction
----------

Beta Skew-t EGARCH models were proposed by Harvey and Chakravarty (2008). They extend on GARCH models through the use of a Skew-t conditional score to drive the conditional variance. This formulation allows for increased robustness to outliers. The basic formulation follows that of a Beta-t-EGARCH model. The Skew-t distribution employed originates from Fernandez and Steel (1998).

.. math::
  
   f\left(\epsilon_{t}\mid\gamma\right) = \frac{2}{\gamma + \gamma^{-1}}\left[{f}\left(\frac{\epsilon_{t}}{\gamma}\right)I_{\left[0,\infty\right]}\left(\epsilon_{t}\right) + f\left(\epsilon_{t}\gamma\right)I_{\left(-\infty,{0}\right)}\left(\epsilon_{t}\right)\right]

The :math:`\gamma` latent variable represents the degree of skewness; for :math:`\gamma=1`, there is no skewness, for :math:`\gamma>1` there is positive skewness, and for :math:`\gamma<1` there is negative skewness.

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

   jpm = DataReader('JPM',  'yahoo', datetime(2006,1,1), datetime(2016,3,10))
   returns = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
   returns.index = jpm.index.values[1:jpm.index.values.shape[0]]
   returns.columns = ['JPM Returns']

   plt.figure(figsize=(15,5));
   plt.plot(returns.index,returns);
   plt.ylabel('Returns');
   plt.title('JPM Returns');

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_7_1.png

One way to visualize the underlying volatility of the series is to plot the absolute returns :math:`\mid{y}\mid`: 

.. code-block:: python

   plt.figure(figsize=(15,5))
   plt.plot(returns.index, np.abs(returns))
   plt.ylabel('Absolute Returns')
   plt.title('JP Morgan Absolute Returns');

.. image:: http://www.pyflux.com/notebooks/GARCH/output_14_02.png

There appears to be some evidence of volatility clustering over this period. Let’s fit a :math:`Beta` :math:`Skew-t` :math:`EGARCH(1,1)` model using a point mass estimate :math:`z^{MLE}`:

.. code-block:: python
   
   skewt_model = pf.SEGARCH(p=1, q=1, data=returns, target='JPM Returns')
   x = skewt_model.fit()
   x.summary()

   SEGARCH(1,1)                                                                                              
   ======================================== =================================================
   Dependent Variable: JPM Returns          Method: MLE                                       
   Start Date: 2006-01-05 00:00:00          Log Likelihood: 6664.2692                         
   End Date: 2016-03-10 00:00:00            AIC: -13316.5384                                  
   Number of observations: 2562             BIC: -13281.4472                                  
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Vol Constant              -0.0586    0.0249     -2.3589  0.0183   (-0.1073 | -0.0099)      
   p(1)                      0.9932                                                           
   q(1)                      0.104                                                            
   Skewness                  0.9858                                                           
   v                         6.0465                                                           
   Returns Constant          0.0015     0.0057     0.271    0.7864   (-0.0096 | 0.0127)       
   ==========================================================================================

The standard errors are not shown for transformed variables. You can pass through a ``transformed=False`` argument to ``summary`` to obtain this information for untransformed variables.

We can plot the skewness latent variable :math:`\gamma` with :py:func:`plot_z`: 

.. code-block:: python

   skewt_model.plot_z([3],figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_11_1.png

So the series is slightly negatively skewed – which is consistent with the direction of skewness for most financial time series. We can plot the fit with :py:func:`plot_fit`: 

.. code-block:: python

   skewt_model.plot_fit(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_13_0.png

And plot predictions of future conditional volatility with :py:func:`plot_predict`: 

.. code-block:: python

   model.plot_predict(h=10)

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_15_0.png

If we had wanted predictions in dataframe form, we could have used :py:func:`predict`: instead. 

We can also estimate a Beta-t-EGARCH model with leverage through :py:func:`add_leverage`: 

.. code-block:: python

   skewt_model = pf.SEGARCH(p=1,q=1,data=returns,target='JPM Returns')
   skewt_model.add_leverage()
   x = skewt_model.fit()
   x.summary()

   SEGARCH(1,1)                                                                                              
   ======================================== =================================================
   Dependent Variable: JPM Returns          Method: MLE                                       
   Start Date: 2006-01-05 00:00:00          Log Likelihood: 6684.9381                         
   End Date: 2016-03-10 00:00:00            AIC: -13355.8762                                  
   Number of observations: 2562             BIC: -13314.9364                                  
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Vol Constant              -0.1203    0.0152     -7.898   0.0      (-0.1501 | -0.0904)      
   p(1)                      0.9857                                                           
   q(1)                      0.1097                                                           
   Leverage Term             0.0713     0.0095     7.5284   0.0      (0.0527 | 0.0899)        
   Skewness                  0.9984                                                           
   v                         5.9741                                                           
   Returns Constant          0.0004     0.0001     6.9425   0.0      (0.0003 | 0.0006)        
   ==========================================================================================

We have a small leverage effect for the time series. We can plot the fit:

.. code-block:: python

   skewt_model.plot_fit(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_19_0.png

And we can plot ahead with the new model:

.. code-block:: python

   skewt_model.plot_predict(h=30,figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/SkewtEGARCH/output_21_0.png


Class Description
----------

.. py:class:: SEGARCH(data, p, q, target)

   **Beta Skew-t EGARCH Models**

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

Black, F. (1976) Studies of stock price volatility changes. In: Proceedings of the 1976 Meetings
of the American Statistical Association. pp. 171–181.

Fernandez, C., & Steel, M. F. J. (1998a). On Bayesian Modeling of Fat Tails and
Skewness. Journal of the American Statistical Association, 93, 359–371.

Harvey, A.C. & Chakravarty, T. (2008) Beta-t-(E)GARCH. Cambridge Working Papers in Economics 0840,
Faculty of Economics, University of Cambridge, 2008. [p137]

Harvey, A.C. & Sucarrat, G. (2013) EGARCH models with fat tails, skewness and leverage. Computational
Statistics and Data Analysis, Forthcoming, 2013. URL http://dx.doi.org/10.1016/j.csda.2013.09.
022. [p138, 139, 140, 143]

Nelson, D. B. (1991), ‘Conditional heteroskedasticity in asset returns: A new
approach’, Econometrica 59, 347—370.