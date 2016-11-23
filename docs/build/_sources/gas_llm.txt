GAS local level models
==================================

Introduction
----------

The principle behind score-driven models is that the linear update :math:`y_{t} - \theta_{t}`, that the Kalman filter relies upon, can be robustified by replacing it with the conditional score of a non-normal distribution. For this reason, any class of traditional state space model has a score-driven equivalent.

For example, consider a local level model in this framework:

.. math::

   p\left(y_{t}\mid\theta_{t}\right)

.. math::

   \theta_{t} = \theta_{t-1} + \eta{H_{t-1}^{-1}S_{t-1}}

Here the underlying parameter :math:`\theta_{t}` resembles a random walk, while the score :math:`S_{t-1}` drives the equation. The first term :math:`H_{t-1}` is a scaling term that can incorporate second-order information, as per a Newton update. The term :math:`\eta` is a learning rate or scaling term, and is the latent variable which is estimated in the model.

Example
----------

We will use data on the number of goals scored by soccer teams Nottingham Forest and Derby in their head-to-head matches from the beginning of their competitive history. We are interested to know whether these games have become more or less high scoring over time.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import pyflux as pf
   import matplotlib.pyplot as plt
   %matplotlib inline

   eastmidlandsderby = pd.read_csv('http://www.pyflux.com/notebooks/eastmidlandsderby.csv')
   total_goals = pd.DataFrame(eastmidlandsderby['Forest'] + eastmidlandsderby['Derby'])
   total_goals.columns = ['Total Goals']
   plt.figure(figsize=(15,5))
   plt.title("Total Goals in the East Midlands Derby")
   plt.xlabel("Games Played");
   plt.ylabel("Total Goals");
   plt.plot(total_goals);

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/output_10_0.png

Here can fit a GAS Local Level model with a ``Poisson()`` family:

.. code-block:: python
   
   model = pf.GASLLEV(data=total_goals, family=pf.Poisson())

Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit("MLE")
   x.summary()

   Poisson GAS LLM                                                                                           
   ======================================== =================================================
   Dependent Variable: Total Goals          Method: MLE                                       
   Start Date: 1                            Log Likelihood: -198.7993                         
   End Date: 96                             AIC: 399.5985                                     
   Number of observations: 96               BIC: 402.1629                                     
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   SC(1)                     0.1929     0.0468     4.1252   0.0      (0.1013 | 0.2846)        
   ==========================================================================================

We can plot the in-sample fit using :py:func:`plot_fit`: 

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/output_14_0.png

If we want to plot predictions, we can use the :py:func:`plot_predict`: method: 

.. code-block:: python

   model.plot_predict(h=10, past_values=30, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/GASStateSpace/output_16_0.png

If we want the predictions in a DataFrame form, then we can just use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: GASLLEV(data, integ, target, family)

   **GAS Local Level Models.**

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

