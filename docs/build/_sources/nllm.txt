Non-Gaussian Local Level Models
==================================

Introduction
----------

With Non-Gaussian state space models, we have the same basic setup as Gaussian state space models, but now a potentially non-Gaussian measurement density. That is we are interested in problems of the form:

.. math::
   
   p\left(y_{t}\mid{z}_{t}\right)

.. math::
   
   \theta_{t} = f\left(\alpha_{t}\right)

.. math::
   
   \alpha_{t} =  \alpha_{t-1} + \eta_{t}

.. math::
   
   \eta_{t} \sim N\left(0,\Sigma\right)

Usually MCMC based schemes are the right way to tackle this problem. Currently PyFlux uses BBVI for speed, but the mean-field approximation means there can be some bias in the states (although the results are generally okay for prediction). In the future, PyFlux will use a more structured approximation.

The **Non-Gaussian local level model** has the same form as a Gaussian local level model, but with a non-Gaussian measurement density.

Example
----------

For fun, and since it’s topical, we’ll apply a Poisson local level model to count data on the number of goals the football team Leicester have scored since they rejoined the Premier League. Each index represents a match they have played. This is a short dataset, but it shows the principle behind the model.

.. code-block:: python

   import numpy as np
   import pyflux as pf
   import pandas as pd
   import matplotlib.pyplot as plt
   %matplotlib inline 

   leicester = pd.read_csv('http://www.pyflux.com/notebooks/leicester_goals_scored.csv')
   leicester.columns= ["Time","Goals","Season2"]
   plt.figure(figsize=(15,5))
   plt.plot(leicester["Goals"])
   plt.ylabel('Goals Scored')
   plt.title('Leicester Goals Since Joining EPL');
   plt.show()

.. image:: http://www.pyflux.com/notebooks/NonGaussianStateSpace/output_11_0.png

We can fit a Poisson local level model as follows:

.. code-block:: python

   model = pf.NLLEV(data=leicester, target='Goals', family=pf.Poisson())

We can also use the higher-level wrapper which allows us to specify the family. If you pick a Normal distribution, then the Kalman filter will be used:

.. code-block:: python
   
   model = pf.LocalLevel(data=leicester, target='Goals', family=pf.Poisson())
 
Next we estimate the latent variables through a BBVI estimate :math:`z^{BBVI}`: 

.. code-block:: python

   x = model.fit(iterations=5000)
   x.summary()

   10% done : ELBO is -107.599165657
   20% done : ELBO is -127.571498111
   30% done : ELBO is -136.25857363
   40% done : ELBO is -137.626516299
   50% done : ELBO is -137.539662707
   60% done : ELBO is -137.321490055
   70% done : ELBO is -137.518451697
   80% done : ELBO is -137.311382466
   90% done : ELBO is -136.3580387
   100% done : ELBO is -137.346927749

   Final model ELBO is -135.76799195

   Poisson Local Level Model                                                                                 
   ======================================== =================================================
   Dependent Variable: Goals                Method: BBVI                                      
   Start Date: 0                            Unnormalized Log Posterior: -56.8409              
   End Date: 74                             AIC: 115.681720125                                
   Number of observations: 75               BIC: 117.999208239                                
   ==========================================================================================
   Latent Variable           Median             Mean               95% Credibility Interval 
   ========================= ================== ================== ==========================
   Sigma^2 level             0.0406             0.0406             (0.0353 | 0.0467)        
   ==========================================================================================

We can plot the evolution parameter with :py:func:`plot_z`:

.. code-block:: python
   
   model.plot_z()

.. image:: http://www.pyflux.com/notebooks/NonGaussianStateSpace/output_15_1.png

Next we will plot the in-sample fit using :py:func:`plot_fit`:

.. code-block:: python

   model.plot_fit(figsize=(15,10))

.. image:: http://www.pyflux.com/notebooks/NonGaussianStateSpace/output_17_0.png

The sharp changes at the beginning reflect the diffuse initialization; together with high initial uncertainty, this leads to stronger updates towards the beginning of the series. We can predict forward using plot_predict: 

We can get an idea of the performance of our model by prediction through the :py:func:`plot_predict`: method:

.. code-block:: python

   model.plot_predict(h=5,figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/NonGaussianStateSpace/output_19_0.png

If we just want the predictions themselves, we can use the :py:func:`predict`: method.

Class Description
----------

.. py:class:: NLLEV(data, ar, integ, target, family)

   **Non-Gaussian Local Level Models (NLLEV).**

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

References
----------

Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. 
Cambridge University Press, Cambridge.

Ranganath, R., Gerrish, S., and Blei, D. M. (2014). Black box variational inference. 
In Artificial Intelligence and Statistics.