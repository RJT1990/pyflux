ARIMA models
==================================

Introduction
----------

Autoregressive integrated moving average (ARIMA) models were popularised by Box and Jenkins (1970). An ARIMA model describes a univariate time series as a combination of autoregressive (AR) and moving average (MA) lags which capture the autocorrelation within the time series. The order of integration denotes how many times the series has been differenced to obtain a stationary series. 

We write an :math:`ARIMA(p,d,q)` model for some time series data :math:`y_{t}`, where :math:`p` is the number of autoregressive lags, :math:`d` is the degree of differencing and :math:`q` is the number of moving average lags as:

.. math::

   \Delta^{D}y_{t} = \sum^{p}_{i=1}\phi_{i}\Delta^{D}y_{t-i} + \sum^{q}_{j=1}\theta_{j}\epsilon_{t-j} + \epsilon_{t}

.. math::

   \epsilon_{t} \sim N\left(0,\sigma^{2}\right)

ARIMA models are associated with a Box-Jenkins approach to time series. According to this approach, you should difference the series until it is stationary, and then use information criteria and autocorrelation plots to choose the appropriate lag order for an :math:`ARIMA` process. You then apply inference to obtain latent variable estimates, and check the model to see whether the model has captured the autocorrelation in the time series. For example, you can plot the autocorrelation of the model residuals. Once you are happy, you can use the model for retrospection and forecasting.

Example
----------

We’ll run an ARIMA Model for yearly sunspot data. First we load the data:

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

.. image:: http://www.pyflux.com/notebooks/ARIMA/output_8_0.png

We can build an ARIMA model as follows, specifying the order of model we want, as well as a pandas DataFrame or numpy array carrying the data. Here we specify an arbitrary :math:`ARIMA(4,0,4)` model: 

.. code-block:: python
   
   model = pf.ARIMA(data=data, ar=4, ma=4, target='sunspot.year', family=pf.Normal())

Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit("MLE")
   x.summary()

   ARIMA(4,0,4)               
   ======================================== ============================================
   Dependent Variable: sunspot.year         Method: MLE                                       
   Start Date: 1704                         Log Likelihood: -1189.488                         
   End Date: 1988                           AIC: 2398.9759                                    
   Number of observations: 285              BIC: 2435.5008                                    
   =====================================================================================
   Latent Variable      Estimate   Std Error  z        P>|z|    95% C.I.                 
   ==================== ========== ========== ======== ======== ========================
   Constant             8.0092     3.2275     2.4816   0.0131   (1.6834 | 14.3351)       
   AR(1)                1.6255     0.0367     44.2529  0.0      (1.5535 | 1.6975)        
   AR(2)                -0.4345    0.2455     -1.7701  0.0767   (-0.9157 | 0.0466)       
   AR(3)                -0.8819    0.2295     -3.8432  0.0001   (-1.3317 | -0.4322)      
   AR(4)                0.5261     0.0429     12.2515  0.0      (0.4419 | 0.6103)        
   MA(1)                -0.5061    0.0383     -13.2153 0.0      (-0.5812 | -0.4311)      
   MA(2)                -0.481     0.1361     -3.533   0.0004   (-0.7478 | -0.2142)      
   MA(3)                0.2511     0.1093     2.2979   0.0216   (0.0369 | 0.4653)        
   MA(4)                0.2846     0.0602     4.7242   0.0      (0.1665 | 0.4027)        
   Sigma                15.7944                                                          
   =====================================================================================


We can plot the latent variables :math:`z^{MLE}`: using the :py:func:`plot_z`: method:

.. code-block:: python

   model.plot_z(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/ARIMA/output_14_0.png

We can plot the in-sample fit using :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/ARIMA/output_16_0.png

We can get an idea of the performance of our model by using rolling in-sample prediction through the :py:func:`plot_predict_is`: method:

.. code-block:: python

   model.plot_predict_is(h=50, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/ARIMA/output_18_0.png

If we want to plot predictions, we can use the :py:func:`plot_predict`: method: 

.. code-block:: python

   model.plot_predict(h=20,past_values=20,figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/ARIMA/output_20_0.png

If we want the predictions in a DataFrame form, then we can just use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: ARIMA(data, ar, ma, integ, target, family)

   **Autoregressive Integrated Moving Average Models (ARIMA).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   ar                   int                                The number of autoregressive lags
   ma                   int                                The number of moving average lags
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

Box, G; Jenkins, G. (1970). Time Series Analysis: Forecasting and Control. San Francisco: Holden-Day.

Hamilton, J.D. (1994). Time Series Analysis. Taylor & Francis US.

