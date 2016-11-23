GAS models
==================================

Introduction
----------

Generalized Autoregressive Score (GAS) models are a recent class of observation-driven time-series model for non-normal data, introduced by Creal et al. (2013) and Harvey (2013). For a conditional observation density :math:`p\left(y_{t}\mid{x_{t}}\right)` with an observation :math:`y_{t}` and a latent time-varying parameter :math:`x_{t}`, we assume the parameter :math:`x_{t}` follows the recursion: 

.. math::

   x_{t} = \mu + \sum^{p}_{i=1}\phi_{i}x_{t-i} + \sum^{q}_{j=1}\alpha_{j}S\left(x_{j-1}\right)\frac{\partial\log p\left(y_{t-j}\mid{x_{t-j}}\right) }{\partial{x_{t-j}}}

For example, for a Poisson distribution density, where the default scaling is :math:`\exp\left(x_{j}\right)`, the time-varying parameter follows:

.. math::

   x_{t} = \mu + \sum^{p}_{i=1}\phi_{i}x_{t-i} + \sum^{q}_{j=1}\alpha_{j}\left(\frac{y_{t-j}}{\exp\left(x_{t-j}\right)} - 1\right)

These types of model can be viewed as approximations to parameter-driven state space models, and are often competitive in predictive performance. See **GAS State Space models** for a more general class of models that extend beyond the simple autoregressive form. The simple GAS models considered here in this notebook can be viewed as an approximation to non-linear ARIMA processes.

Example
----------

We demonstrate an example below for count data. The data below records if a country somewhere in the world experiences a banking crisis in a given year.

.. code-block:: python

   import numpy as np
   import pyflux as pf
   import pandas as pd
   import matplotlib.pyplot as plt
   %matplotlib inline 

   data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/bankingCrises.csv")
   numpy_data = np.sum(data.iloc[:,2:73].values,axis=1)
   numpy_data[np.isnan(numpy_data)] = 0
   financial_crises = pd.DataFrame(numpy_data)
   financial_crises.index = data.year
   financial_crises.columns = ["Number of banking crises"]

   plt.figure(figsize=(15,5))
   plt.plot(financial_crises)
   plt.ylabel("Count")
   plt.xlabel("Year")
   plt.title("Number of banking crises across the world")
   plt.show()

.. image:: http://www.pyflux.com/notebooks/GAS/output_13_0.png

Here we specify an arbitrary :math:`GAS(2,0,2)` model with a ``Poisson()`` family:

.. code-block:: python
   
   model = pf.GAS(ar=2, sc=2, data=financial_crises, family=pf.Poisson())

Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit("MLE")
   x.summary()

   PoissonGAS (2,0,2)                                                                                        
   ======================================== ==================================================
   Dependent Variable: No of crises         Method: MLE                                       
   Start Date: 1802                         Log Likelihood: -497.8648                         
   End Date: 2010                           AIC: 1005.7297                                    
   Number of observations: 209              BIC: 1022.4413                                    
   ===========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== =========================
   Constant                  0.0        0.0267     0.0      1.0      (-0.0524 | 0.0524)       
   AR(1)                     0.4502     0.0552     8.1544   0.0      (0.342 | 0.5584)         
   AR(2)                     0.4595     0.0782     5.8776   0.0      (0.3063 | 0.6128)        
   SC(1)                     0.2144     0.0241     8.8929   0.0      (0.1671 | 0.2616)        
   SC(2)                     0.0571     0.0042     13.5323  0.0      (0.0488 | 0.0654)        
   ===========================================================================================

We can plot the latent variables :math:`z^{MLE}`: using the :py:func:`plot_z`: method:

.. code-block:: python

   model.plot_z(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GAS/output_17_0.png

We can plot the in-sample fit using :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/GAS/output_19_0.png

We can get an idea of the performance of our model by using rolling in-sample prediction through the :py:func:`plot_predict_is`: method:

.. code-block:: python

   model.plot_predict_is(h=20, fit_once=True, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GAS/output_21_0.png

If we want to plot predictions, we can use the :py:func:`plot_predict`: method: 

.. code-block:: python

   model.plot_predict(h=10, past_values=30, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GAS/output_23_0.png

If we want the predictions in a DataFrame form, then we can just use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: GAS(data, ar, sc, integ, target, family)

   **Generalized Autoregressive Score Models (GAS).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   ar                   int                                The number of autoregressive lags
   sc                   int                                The number of score function lags
   integ                int                                How many times to difference the data
                                                           (default: 0)
   target               string or int                      Which column of DataFrame/array to use.
   family               pf.Family instance                 The distribution for the time series,
                                                           e.g ``pf.Normal()``
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

Creal, D; Koopman, S.J.; Lucas, A. (2013). Generalized Autoregressive Score Models with
Applications. Journal of Applied Econometrics, 28(5), 777–795. doi:10.1002/jae.1279.

Harvey, A.C. (2013). Dynamic Models for Volatility and Heavy Tails: With Applications to
Financial and Economic Time Series. Cambridge University Press.

