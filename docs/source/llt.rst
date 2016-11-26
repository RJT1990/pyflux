Gaussian Local Linear Trend models
==================================

Introduction
----------

Gaussian state space models - often called structural time series or unobserved component models - provide a way to decompose a time series into several distinct components. These components can be extracted in closed form using the Kalman filter if the errors are jointly Gaussian, and parameters can be estimated via the prediction error decomposition and Maximum Likelihood.

One classic univariate structural time series model is the **local linear trend model**. We can write this as a combination of a time-varying level, time-varying trend and an irregular term:

.. math::

   y_{t} = \mu_{t} + \epsilon_{t}

.. math::

   \mu_{t} =  \beta_{t-1} + \mu_{t-1} + \eta_{t}

.. math::

   \beta_{t} =  \beta_{t-1} + \zeta_{t}

.. math::

   \epsilon_{t} \sim N\left(0,\sigma_{\epsilon}^{2}\right)

.. math::

   \eta_{t} \sim N\left(0,\sigma_{\eta}^{2}\right)

.. math::

   \zeta_{t} \sim N\left(0,\sigma_{\zeta}^{2}\right)


Example
----------

We will use data on logged US Real GDP.

.. code-block:: python

   growthdata = pd.read_csv('http://www.pyflux.com/notebooks/GDPC1.csv')
   USgrowth = pd.DataFrame(np.log(growthdata['VALUE']))
   USgrowth.index = pd.to_datetime(growthdata['DATE'])
   USgrowth.columns = ['Logged US Real GDP']
   plt.figure(figsize=(15,5))
   plt.plot(USgrowth.index, USgrowth)
   plt.ylabel('Real GDP')
   plt.title('US Real GDP');

.. image:: http://www.pyflux.com/notebooks/GaussianStateSpace/output_27_0.png

Here define a Local Linear Trend model as follows:

.. code-block:: python
   
   model = pf.LLT(data=USgrowth)

We can also use the higher-level wrapper which allows us to specify the family, although if we pick a non-Gaussian family then the model will be estimated in a different way (not through the Kalman filter):

.. code-block:: python
   
   model = pf.LocalTrend(data=USgrowth, family=pf.Normal())

Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit()
   x.summary()

   LLT                                                                                                       
   ======================================== =================================================
   Dependent Variable: Logged US Real GDP   Method: MLE                                       
   Start Date: 1947-01-01 00:00:00          Log Likelihood: 877.3334                          
   End Date: 2015-10-01 00:00:00            AIC: -1748.6667                                   
   Number of observations: 276              BIC: -1737.8055                                   
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Sigma^2 irregular         3.306e-07                                                        
   Sigma^2 level             5.41832e-0                                                       
   Sigma^2 trend             1.55224e-0                                                       
   ==========================================================================================

We can plot the in-sample fit using :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/GaussianStateSpace/output_31_0.png

The trend picks out the underlying local growth rate in the series. Together with the local level this constitutes the smoothed series. We can use the simulation smoother to simulate draws from the states, using py:func:`simulation_smoother`:. We draw from the local trend state: 

.. code-block:: python

   plt.figure(figsize=(15,5))
   for i in range(10):
       plt.plot(model.index,model.simulation_smoother(
               model.latent_variables.get_z_values())[1][0:model.index.shape[0]])
   plt.show()

.. image:: http://www.pyflux.com/notebooks/GaussianStateSpace/output_33_0.png

If we want to plot rolling in-sample predictions, we can use the :py:func:`plot_predict_is`: method: 

.. code-block:: python

   model.plot_predict_is(h=15,figsize=(15,5))

.. image::  http://www.pyflux.com/notebooks/GaussianStateSpace/output_35_0.png

We can view out-of-sample predictions using :py:func:`plot_predict`:

.. code-block:: python

   model.plot_predict(h=20, past_values=17*4, figsize=(15,6))

.. image::  http://www.pyflux.com/notebooks/GaussianStateSpace/output_37_0.png

If we want the predictions in a DataFrame form, then we can just use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: LLT(data, integ, target)

   **Local Linear Trend Models.**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   integ                int                                How many times to difference the data
                                                           (default: 0)
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
