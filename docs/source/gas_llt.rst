GAS local linear trend models
==================================

Introduction
----------

The principle behind score-driven models is that the linear update :math:`y_{t} - \theta_{t}`, that the Kalman filter relies upon, can be robustified by replacing it with the conditional score of a non-normal distribution. For this reason, any class of traditional state space model has a score-driven equivalent.

For example, consider a local linear model in this framework:

.. math::

   p\left(y_{t}\mid\mu_{t}\right)

.. math::

   \mu_{t} = \mu_{t-1} + \beta_{t-1} + \eta_{1}{H_{t-1}^{-1}S_{t-1}}

.. math::

   \beta_{t} = \beta_{t-1} + \eta_{2}{H_{t-1}^{-1}S_{t-1}}

Here :math:`\eta` represents the two learning rates or scaling terms, and are the latent variables which are estimated in the model.

Example
----------

We will construct a local linear trend model for US GDP using a skew t-distribution. Here is the data: 

.. code-block:: python

   growthdata = pd.read_csv('http://www.pyflux.com/notebooks/GDPC1.csv')
   USgrowth = pd.DataFrame(np.log(growthdata['VALUE']))
   USgrowth.index = pd.to_datetime(growthdata['DATE'])
   USgrowth.columns = ['Logged US Real GDP']
   plt.figure(figsize=(15,5))
   plt.plot(USgrowth.index, USgrowth)
   plt.ylabel('Real GDP')
   plt.title('US Logged Real GDP');

.. image:: http://www.pyflux.com/notebooks/GASLLT/output_0_0.png

Here can fit a GAS Local Linear Trend model with a ``Skewt()`` family:

.. code-block:: python
   
   model = pf.GASLLT(data=USgrowth-np.mean(USgrowth),family=pf.Skewt())

Next we estimate the latent variables. For this example we will use a BBVI estimate :math:`z^{BBVI}`: 

.. code-block:: python

   x = model.fit('BBVI', iterations=20000, record_elbo=True)
   10% done : ELBO is 70.1403732024, p(y,z) is 82.4582067151, q(z) is 12.3178335126
   20% done : ELBO is 71.5399641383, p(y,z) is 84.3269580596, q(z) is 12.7869939213
   30% done : ELBO is 95.3663747496, p(y,z) is 108.551290696, q(z) is 13.1849159469
   40% done : ELBO is 124.357073241, p(y,z) is 138.132000673, q(z) is 13.7749274322
   50% done : ELBO is 144.111819073, p(y,z) is 158.386802182, q(z) is 14.274983109
   60% done : ELBO is 164.792526642, p(y,z) is 179.422645151, q(z) is 14.6301185085
   70% done : ELBO is 178.18148403, p(y,z) is 193.190633108, q(z) is 15.0091490782
   80% done : ELBO is 206.095112618, p(y,z) is 221.579871841, q(z) is 15.4847592232
   90% done : ELBO is 210.854594358, p(y,z) is 226.705793141, q(z) is 15.8511987823
   100% done : ELBO is 226.965067448, p(y,z) is 243.29536546, q(z) is 16.3302980111

   Final model ELBO is 224.286972026

We can plot the ELBO with :py:func:`plot_elbo`: on the results object:

.. code-block:: python

   x.plot_elbo(figsize=(15,7))

.. image:: http://www.pyflux.com/notebooks/GASLLT/output_2_0.png

We can plot the latent variables with :py:func:`plot_z`: 

.. code-block:: python

   model.plot_z([0,1,3])

.. image:: http://www.pyflux.com/notebooks/GASLLT/output_3_0.png

.. code-block:: python

   model.plot_z([2,4])

.. image:: http://www.pyflux.com/notebooks/GASLLT/output_4_0.png

The states are stored as an attribute ``states`` in the results object. Let's plot the trend state:

.. code-block:: python

   plt.figure(figsize=(15,5))
   plt.title("Local Trend for US GDP")
   plt.ylabel("Trend")
   plt.plot(USgrowth.index[21:],x.states[1][20:]);

.. image:: http://www.pyflux.com/notebooks/GASLLT/output_5_0.png

This reflects the underlying growth potential of the US economy.

We can also calculate the average growth rate for a forward forecast:

.. code-block:: python

   print("Average growth rate for this period is")
   print(str(round(100*np.mean(np.exp(np.diff(model.predict(h=4)['Logged US Real GDP'].values)) - 1),3)) + "%")

   Average growth rate for this period is
   0.504%

Class Description
----------

.. py:class:: GASLLT(data, integ, target, family)

   **GAS Local Linear Trend Models.**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
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

