DAR models
==================================

Introduction
----------

Gaussian state space models - often called structural time series or unobserved component models - provide a way to decompose a time series into several distinct components. These components can be extracted in closed form using the Kalman filter if the errors are jointly Gaussian, and parameters can be estimated via the prediction error decomposition and Maximum Likelihood.

We can write a **dynamic autoregression model** in this framework as:

.. math::

   y_{t} = \sum^{p}_{i=1}\phi_{i,t}y_{t-i} + \epsilon_{t}

.. math::

   \phi_{i,t}= \phi_{i,t-1} + \eta_{i,t}

.. math::

   \epsilon_{t} \sim N\left(0,\sigma^{2}\right)

.. math::

   \eta_{i,t} \sim N\left(0,\sigma_{\eta_{i}}^{2}\right)

In other words the dynamic autoregression coefficients follow a random walk.

Example
----------

We’ll run an Dynamic Autoregressive (DAR) Model for yearly sunspot data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import pyflux as pf
   from datetime import datetime
   import matplotlib.pyplot as plt
   %matplotlib inline 

   data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv')
   data.index = data['time'].values

   plt.figure(figsize=(15,5))
   plt.plot(data.index,data['sunspot.year'])
   plt.ylabel('Sunspots')
   plt.title('Yearly Sunspot Data');

.. image:: http://www.pyflux.com/notebooks/DAR/output_7_0.png

Here we specify an arbitrary DAR(9) model (note: which is probably overspecified).

.. code-block:: python

   model = pf.DAR(data=data, ar=9, integ=0, target='sunspot.year')
 
Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit("MLE")
   x.summary()

   DAR(9, integrated=0)                                                                                      
   ====================================== =================================================
   Dependent Variable: sunspot.year       Method: MLE                                       
   Start Date: 1709                       Log Likelihood: -1179.097                         
   End Date: 1988                         AIC: 2380.194                                     
   Number of observations: 280            BIC: 2420.1766                                    
   ========================================================================================
   Latent Variable         Estimate   Std Error  z        P>|z|    95% C.I.                 
   ======================= ========== ========== ======== ======== ========================
   Sigma^2 irregular       0.301                                                            
   Constant                60.0568    23.83      2.5202   0.0117   (13.3499 | 106.7637)     
   Sigma^2 AR(1)           0.005                                                            
   Sigma^2 AR(2)           0.0                                                              
   Sigma^2 AR(3)           0.0005                                                           
   Sigma^2 AR(4)           0.0001                                                           
   Sigma^2 AR(5)           0.0002                                                           
   Sigma^2 AR(6)           0.0011                                                           
   Sigma^2 AR(7)           0.0002                                                           
   Sigma^2 AR(8)           0.0003                                                           
   Sigma^2 AR(9)           0.032                                                            
   =========================================================================================

Note we have no standard errors in the results table because it shows the transformed parameters. If we want standard errors, we can call ``x.summary(transformed=False)``. Next we will plot the in-sample fit and the dynamic coefficients using :py:func:`plot_fit`:

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/DAR/output_13_0.png

The sharp changes at the beginning reflect the diffuse initialization; together with high initial uncertainty, this leads to stronger updates towards the beginning of the series. We can predict forward using plot_predict: 

We can predict forwards through the :py:func:`plot_predict`: method:

.. code-block:: python

   model.plot_predict(h=50, past_values=40, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/DAR/output_15_0.png

The prediction intervals here are unrealistic and reflect the Gaussian distributional assumption we’ve chosen – we can’t have negative sunspots! – but if we are just want the predictions themselves, we can use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: DAR(data, ar, integ, target, family)

   **Dynamic Autoregression Models (DAR).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   ar                   int                                The number of autoregressive lags
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

   .. py:method:: plot_z(indices, figsize)

      Returns a plot of the latent variables and their associated uncertainty. 

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      indices              int or list                 Which latent variable indices to plot
      figsize              tuple                       Size of the matplotlib figure
      ==================   ========================    ======================================

      **Returns** : void - shows a matplotlib plot

   .. py:method:: predict(h)
      
      Returns a DataFrame of model predictions.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many steps to forecast ahead
      ==================   ========================    ======================================

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

   .. py:method:: simulation_smoother(beta)

      Returns np.ndarray of draws of the data from the Durbin and Koopman (2002) simulation smoother.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      beta                 np.array                    np.array of latent variables
      ==================   ========================    ======================================

      Recommended just to use model.latent_variables.get_z_values() for the beta input, if you
      have already fit a model.

      **Returns** : np.ndarray - samples from simulation smoother 

References
----------

Durbin, J. and Koopman, S. J. (2002). A simple and efficient simulation smoother for state
space time series analysis. Biometrika, 89(3):603–615.

Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. 
Cambridge University Press, Cambridge.
