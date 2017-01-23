Dynamic Linear regression models
==================================

Introduction
----------

Gaussian state space models - often called structural time series or unobserved component models - provide a way to decompose a time series into several distinct components. These components can be extracted in closed form using the Kalman filter if the errors are jointly Gaussian, and parameters can be estimated via the prediction error decomposition and Maximum Likelihood.

We can support **Dynamic Linear Regression** in the state space framework:

.. math::

   y_{t} = \boldsymbol{x}_{t}^{'}\boldsymbol{\beta}_{t} + \epsilon_{t}

.. math::

   \boldsymbol{\beta}_{t} =  \boldsymbol{\beta}_{t-1} + \boldsymbol{\eta}_{t}

.. math::

   \epsilon_{t} \sim N\left(0,\sigma_{\epsilon}^{2}\right)

.. math::

   \boldsymbol{\eta}_{t} \sim N\left(\boldsymbol{0},\Sigma_{\eta}\right)

Example
----------

In constructing portfolios in finance, we are often after the :math:`\beta` of a stock which can be used to construct the systematic component of returns. But this may not be a static quantity. For normally distributed returns (!) we can use a dynamic linear regression model using the Kalman filter and smoothing algorithm to track its evolution. First let’s get some data on excess returns. We’ll look at Amazon stock (AMZN) and use the S&P500 as ‘the market’. 

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

.. image:: http://www.pyflux.com/notebooks/GaussianStateSpace/output_43_1.png

Here we define a Dynamic Linear regression as follows:

.. code-block:: python
   
   model = pf.DynReg('Amazon ~ SP500', data=final_returns)

We can also use the higher-level wrapper which allows us to specify the family, although if we pick a non-Gaussian family then the model will be estimated in a different way (not through the Kalman filter):

.. code-block:: python
   
   model = pf.DynamicGLM('Amazon ~ SP500', data=USgrowth, family=pf.Normal())

Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit()
   x.summary()

   Dynamic Linear Regression                                                                                 
   ====================================== =================================================
   Dependent Variable: Amazon             Method: MLE                                       
   Start Date: 2012-01-04 00:00:00        Log Likelihood: 2871.5419                         
   End Date: 2016-06-01 00:00:00          AIC: -5737.0838                                   
   Number of observations: 1101           BIC: -5722.0719                                   
   ========================================================================================
   Latent Variable         Estimate   Std Error  z        P>|z|    95% C.I.                 
   ======================= ========== ========== ======== ======== ========================
   Sigma^2 irregular       0.0003                                                           
   Sigma^2 1               0.0                                                              
   Sigma^2 SP500           0.0024                                                           
   ========================================================================================

We can plot the in-sample fit using :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,15))

.. image:: http://www.pyflux.com/notebooks/GaussianStateSpace/output_47_0.png

The third plot shows :math:`\beta_{AMZN}`. Following the burn-in period, the :math:`\beta` hovered just above 1 in 2013, although it became very correlated with market performance in 2014. More recently it has settled down again to hover just above 1. The fourth plot shows the remaining residual component of return (not including :math:`\alpha`).

Class Description
----------

.. py:class:: DynLin(formula, data)

   **Dynamic Linear Regression models**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   formula              string                             Patsy notation specifying the regression
   data                 pd.DataFrame                       Contains the univariate time series
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
