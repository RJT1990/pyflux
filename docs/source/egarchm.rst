Beta-t-EGARCH in-mean models
==================================

Introduction
----------

Beta-t-EGARCH in-mean models extend :math:`Beta`-:math:`t`-:math:`EGARCH(p,q)` models by allowing returns to depend upon a past conditional volatility component. The :math:`Beta`-:math:`t`-:math:`EGARCH(p,q)` in-mean model includes this effect :math:`\phi` as follows:

.. math::
  
   y_{t} =  \mu + \phi\exp\left(\lambda_{t\mid{t-1}}/2\right) + \exp\left(\lambda_{t\mid{t-1}}/2\right)\epsilon_{t}

.. math::
  
   \lambda_{t\mid{t-1}} = \alpha_{0} + \sum^{p}_{i=1}\alpha_{i}\lambda_{t-i} + \sum^{q}_{j=1}\beta_{j}u_{t-j}
  
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

   jpm = DataReader('JPM',  'yahoo', datetime(2006,1,1), datetime(2016,3,10))
   returns = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
   returns.index = jpm.index.values[1:jpm.index.values.shape[0]]
   returns.columns = ['JPM Returns']

   plt.figure(figsize=(15,5));
   plt.plot(returns.index,returns);
   plt.ylabel('Returns');
   plt.title('JPM Returns');

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_0_0.png

One way to visualize the underlying volatility of the series is to plot the absolute returns :math:`\mid{y}\mid`: 

.. code-block:: python

   plt.figure(figsize=(15,5))
   plt.plot(returns.index, np.abs(returns))
   plt.ylabel('Absolute Returns')
   plt.title('JP Morgan Absolute Returns');

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_1_0.png

There appears to be some evidence of volatility clustering over this period. Let’s fit a :math:`Beta`-:math:`t`-:math:`EGARCH-M(1,1)` model using a BBVI estimate :math:`z^{BBVI}`:

.. code-block:: python
   
   model = pf.EGARCHM(returns,p=1,q=1)
   x = model.fit('BBVI', record_elbo=True, iterations=1000, map_start=False)
   x.summary()

   EGARCHM(1,1)                                                                                              
   ======================================== ==================================================
   Dependent Variable: AAPL Returns         Method: BBVI                                      
   Start Date: 2006-01-05 00:00:00          Unnormalized Log Posterior: 7016.148              
   End Date: 2016-11-11 00:00:00            AIC: -14020.2959822                               
   Number of observations: 2734             BIC: -13984.8148561                               
   ===========================================================================================
   Latent Variable             Median             Mean               95% Credibility Interval 
   =========================== ================== ================== =========================
   Vol Constant                -0.2842            -0.2841            (-0.3474 | -0.2207)      
   p(1)                        0.9624             0.9624             (0.9579 | 0.9665)        
   q(1)                        0.1889             0.1889             (0.1784 | 0.2)           
   v                           8.689              8.6902             (8.0781 | 9.3552)        
   Returns Constant            0.0001             0.0001             (-0.0094 | 0.0093)       
   GARCH-M                     0.1087             0.1085             (0.0226 | 0.1942)        
   ===========================================================================================

We can plot the ELBO through BBVI by calling :py:func:`plot_elbo`: on the results object:

.. code-block:: python

   x.plot_elbo(figsize=(15,7))

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_3_0.png

As we can see, the ELBO converges after around 200 iterations. We can plot the model fit through :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,7))

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_4_0.png

And plot predictions of future conditional volatility with :py:func:`plot_predict`: 

.. code-block:: python

   model.predict(h=10)

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_5_0.png

We can plot samples from the posterior predictive density through :py:func:`plot_sample`: 

.. code-block:: python

   model.plot_sample(figsize=(15, 7))

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_6_0.png

And we can do posterior predictive checks on discrepancies of interest:

.. code-block:: python

   from scipy.stats import kurtosis
   model.plot_ppc(T=kurtosis,figsize=(15, 7))
   model.plot_ppc(T=np.std,figsize=(15, 7))

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_7_0.png

.. image:: http://www.pyflux.com/notebooks/EGARCHM/output_7_1.png

Here it appears our generated samples generate kurtosis that is slightly lower than the data, and a standard deviation that is slightly higher, but we are not too off in both checks.

Class Description
----------

.. py:class:: EGARCHM(data, p, q, target)

   **Beta-t-EGARCH in-mean Models**

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

Harvey, A.C. & Chakravarty, T. (2008) Beta-t-(E)GARCH. Cambridge Working Papers in Economics 0840,
Faculty of Economics, University of Cambridge, 2008. [p137]

Harvey, A.C. & Sucarrat, G. (2013) EGARCH models with fat tails, skewness and leverage. Computational
Statistics and Data Analysis, Forthcoming, 2013. URL http://dx.doi.org/10.1016/j.csda.2013.09.
022. [p138, 139, 140, 143]

Nelson, D. B. (1991), ‘Conditional heteroskedasticity in asset returns: A new
approach’, Econometrica 59, 347—370.