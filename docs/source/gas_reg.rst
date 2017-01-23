GAS regression models
==================================

Introduction
----------

The principle behind score-driven models is that the linear update :math:`y_{t} - \theta_{t}`, that the Kalman filter relies upon, can be robustified by replacing it with the conditional score of a non-normal distribution. For this reason, any class of traditional state space model has a score-driven equivalent.

For example, consider a dynamic regression model in this framework:

.. math::

   p\left(y_{t}\mid\theta_{t}\right)

.. math::

   \theta_{t} = \boldsymbol{x}_{t}^{'}\boldsymbol{\beta}_{t}

.. math::

   \boldsymbol{\beta}_{t} =  \boldsymbol{\beta}_{t-1} + \boldsymbol{\eta}H_{t-1}^{-1}S_{t-1}

Here :math:`\eta` represents the learning rates or scaling terms, and are the latent variables which are estimated in the model.

Example
----------

We will use a dynamic t regression to extract a dynamic :math:`\beta` for a stock. Using t-distributed errors is more robust than a normality assumption, that could be obtained with a Kalman filter. The :math:`\beta` captures the amount of systematic risk in the stock - i.e. the stock's relationship with the market.

.. code-block:: python

   from pandas_datareader import DataReader
   from datetime import datetime

   a = DataReader('AMZN',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
   a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
   a_returns.index = a.index.values[1:a.index.values.shape[0]]
   a_returns.columns = ["Amazon Returns"]

   spy = DataReader('SPY',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
   spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
   spy_returns.index = spy.index.values[1:spy.index.values.shape[0]]
   spy_returns.columns = ['S&P500 Returns']

   one_mon = DataReader('DGS1MO', 'fred',datetime(2012,1,1), datetime(2016,6,1))
   one_day = np.log(1+one_mon)/365

   returns = pd.concat([one_day,a_returns,spy_returns],axis=1).dropna()
   excess_m = returns["Amazon Returns"].values - returns['DGS1MO'].values
   excess_spy = returns["S&P500 Returns"].values - returns['DGS1MO'].values
   final_returns = pd.DataFrame(np.transpose([excess_m,excess_spy, returns['DGS1MO'].values]))
   final_returns.columns=["Amazon","SP500","Risk-free rate"]
   final_returns.index = returns.index

   plt.figure(figsize=(15,5))
   plt.title("Excess Returns")
   x = plt.plot(final_returns);
   plt.legend(iter(x), final_returns.columns);

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/output_31_1.png

We can fit a GAS Regression model with a ``t()`` family:

.. code-block:: python
   
   model = pf.GASReg('Amazon ~ SP500', data=final_returns, family=pf.t())

Next we estimate the latent variables. For this example we will use a Maximum Likelihood estimate :math:`z^{MLE}`: 

.. code-block:: python
  
   x = model3.fit()
   x.summary()

   t GAS Regression                                                                                          
   ======================================== =================================================
   Dependent Variable: Amazon               Method: MLE                                       
   Start Date: 2012-01-04 00:00:00          Log Likelihood: 3158.435                          
   End Date: 2016-06-01 00:00:00            AIC: -6308.87                                     
   Number of observations: 1101             BIC: -6288.8541                                   
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Scale 1                   0.0                                                              
   Scale SP500               0.0474                                                           
   t Scale                   0.0095                                                           
   v                         2.8518                                                           
   ==========================================================================================

We can plot the fit with :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(intervals=False,figsize=(15,15))

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/output_36_0.png

One of the advantages of using a GASRegression rather than a Kalman filtered Dynamic Linear Regression is that the GASRegression with t errors is more robust to outliers. We do not produce the whole analysis here, but for the same data, the filtered estimates are compared below:

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/gaskalman.png

Class Description
----------

.. py:class:: GASReg(data, formula, target, family)

   **Generalized Autoregressive Score Regression Models (GASReg).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   formula              string                             Patsy notation specifying the regression
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

   .. py:method:: plot_predict(h, oos_data, past_values, intervals, **kwargs)
      
      Plots predictions of the model, along with intervals.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many steps to forecast ahead
      oos_data             pd.DataFrame                Exogenous variables in a frame for h steps
      past_values          int                         How many past datapoints to plot
      intervals            boolean                     Whether to plot intervals or not
      ==================   ========================    ======================================

      To be clear, the *oos_data* argument should be a DataFrame in the same format as the initial
      dataframe used to initialize the model instance. The reason is that to predict future values,
      you need to specify assumptions about exogenous variables for the future. For example, if you
      predict *h* steps ahead, the method will take the h first rows from *oos_data* and take the 
      values for the exogenous variables that you asked for in the patsy formula.

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

   .. py:method:: predict(h, oos_data, intervals=False)
      
      Returns a DataFrame of model predictions.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      h                    int                         How many steps to forecast ahead
      oos_data             pd.DataFrame                Exogenous variables in a frame for h steps
      intervals            boolean                     Whether to return prediction intervals
      ==================   ========================    ======================================

      To be clear, the *oos_data* argument should be a DataFrame in the same format as the initial
      dataframe used to initialize the model instance. The reason is that to predict future values,
      you need to specify assumptions about exogenous variables for the future. For example, if you
      predict *h* steps ahead, the method will take the 5 first rows from *oos_data* and take the 
      values for the exogenous variables that you specified as exogenous variables in the patsy formula.

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
