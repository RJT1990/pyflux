GASX models
==================================

Introduction
----------

GASX models extend GAS models by including exogenous factors :math:`X`. For a conditional observation density :math:`p\left(y_{t}\mid{\theta_{t}}\right)` with an observation :math:`y_{t}` and a latent time-varying parameter :math:`\theta_{t}`, we assume the parameter :math:`\theta_{t}` follows the recursion: 

.. math::

   \theta_{t} = \mu + \sum^{K}_{k=1}\beta_{k}X_{t,k} + \sum^{p}_{i=1}\phi_{i}\theta_{t-i} + \sum^{q}_{j=1}\alpha_{j}S\left(x_{j-1}\right)\frac{\partial\log p\left(y_{t-j}\mid{\theta_{t-j}}\right) }{\partial{\theta_{t-j}}}

For example, for the Poisson family, where the default scaling is :math:`\exp\left(\theta_{t}\right)`, the time-varying latent variable follows:

.. math::

   \theta_{t} = \mu + \sum^{K}_{k=1}\beta_{k}X_{t,k} + \sum^{p}_{i=1}\phi_{i}\theta_{t-i} + \sum^{q}_{j=1}\alpha_{j}\left(\frac{y_{t-j}}{\exp\left(\theta_{t-j}\right)} - 1\right)

The model can be viewed as an approximation to a non-linear ARIMAX model.

Example
----------

Below we estimate the :math:`\beta` for a stock – the systematic (market) component of returns – using a heavy tailed distribution and some short-term autoregressive effects. First let’s load some data: 

.. code-block:: python

   from pandas_datareader.data import DataReader
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

.. image:: http://www.pyflux.com/notebooks/GAS/output_28_12.png

Below we estimate a point mass estimate :math:`z^{MLE}` of the latent variables for a :math:`GASX(1,1)` model: 

.. code-block:: python

   model = pf.GASX(formula="Amazon~SP500",data=final_returns,ar=1,sc=1,family=pf.GASSkewt())
   x = model.fit()
   x.summary()

   Skewt GASX(1,0,1)                                                                                         
   ======================================== =================================================
   Dependent Variable: Amazon               Method: MLE                                       
   Start Date: 2012-01-05 00:00:00          Log Likelihood: 3165.9237                         
   End Date: 2016-06-01 00:00:00            AIC: -6317.8474                                   
   Number of observations: 1100             BIC: -6282.8259                                   
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   AR(1)                     0.0807     0.0202     3.9956   0.0001   (0.0411 | 0.1203)        
   SC(1)                     -0.0       0.0187     -0.0001  0.9999   (-0.0367 | 0.0367)       
   Beta 1                    -0.0005    0.0249     -0.0184  0.9853   (-0.0493 | 0.0484)       
   Beta SP500                1.2683     0.0426     29.7473  0.0      (1.1848 | 1.3519)        
   Skewness                  1.017                                                            
   Skewt Scale               0.0093                                                           
   v                         2.7505                                                           
   ==========================================================================================
   WARNING: Skew t distribution is not well-suited for MLE or MAP inference
   Workaround 1: Use a t-distribution instead for MLE/MAP
   Workaround 2: Use M-H or BBVI inference for Skew t distribution

The results table warns us about using the Skew t distribution. This choice of family can sometimes be unstable, so we may want to opt for a t-distribution instead. But in this case, we seem to have obtained sensible results. We can plot the constant and the GAS latent variables by referencing their indices with :py:func:`plot_z`: 

.. code-block:: python

   model.plot_z(indices=[0,1,2])

.. image:: http://www.pyflux.com/notebooks/GAS/output_32_02.png

Similarly we can plot :math:`\beta`:

.. code-block:: python

   model.plot_z(indices=[3])

.. image:: http://www.pyflux.com/notebooks/GAS/output_34_02.png

Our :math:`\beta_{AMZN}` estimate is above 1.0 (fairly strong systematic risk). Let us plot the model fit and the systematic component of returns with :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/GAS/output_36_02.png

Class Description
----------

.. py:class:: GASX(data, formula, ar, sc, integ, target, family)

   **Generalized Autoregressive Score Exogenous Variable Models (GASX).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   formula              string                             Patsy notation specifying the regression
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