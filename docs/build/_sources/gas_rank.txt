GAS ranking models
==================================

Introduction
----------

GAS ranking models can be used for pairwise comparisons or competitive activities. The :math:`GASRank` model of Taylor, R. (2016) models a point difference between two competitors. The point difference is assumed to follow a particular distribution. For example, suppose that the point difference :math:`\mu_{t}` is normally distributed, then we can model the point difference as the location parameter :math:`\mu`:

.. math::

   \mu_{t} = \delta + \alpha_{t,i} -  \alpha_{t,j}

Where :math:`\delta_{t}` is a 'home advantage' latent variable, :math:`i` and :math:`j` refer to home and away competitors, and :math:`\alpha` contains the team power rankings. The power rankings are modelled as random walk processes between each match:

.. math::

   \alpha_{k,i} = \alpha_{k-1,i} + \eta{U}_{k-1,i}

.. math::

   \alpha_{k,j} = \alpha_{k-1,j} - \eta{U}_{k-1,j}

Where :math:`k` is the game index, :math:`\eta` is a learning rate or scaling parameter to be estimated.

The model can be extended to a two component model where each competitor has two aspects to their 'team'. For example, we might model an NFL team along with the Quarterback power rankings in the same game:

.. math::

   \mu_{t} = \delta + \alpha_{t,i} -  \alpha_{t,j}   + \gamma_{t,i} -  \gamma_{t,j}

Here :math:`\gamma` represents the power ranking of the second component. The secondary component power rankings are modelled as random walk processes between each match:

.. math::

   \gamma_{k,i} = \gamma_{k-1,i} + \eta{U_2}_{k-1,i}

.. math::

   \gamma_{k,j} = \gamma_{k-1,j} - \eta{U_2}_{k-1,j}

Developer Note
----------
- This model type has yet to be cythonized, so performance can be slow.

Example
----------

We will model the point difference in NFL games with a simple model. Here is the data:

.. code-block:: python

   import pyflux as pf
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   data = pd.read_csv("nfl_data_new.csv")
   data["PointsDiff"] = data["HomeScore"] - data["AwayScore"]
   data.columns
   Index(['Unnamed: 0', 'AFirstDowns', 'AFumbles0', 'AFumbles1', 'AIntReturns0',
       'AIntReturns1', 'AKickoffReturns0', 'AKickoffReturns1', 'ANetPassYards',
       'APassesAttempted', 'APassesCompleted', 'APassesIntercepted',
       'APenalties0', 'APenalties1', 'APossession', 'APuntReturns0',
       'APuntReturns1', 'APunts0', 'APunts1', 'AQB', 'ARushes0', 'ARushes1',
       'ASacked0', 'ASacked1', 'AwayScore', 'AwayTeam', 'Date', 'HFirstDowns',
       'HFumbles0', 'HFumbles1', 'HIntReturns0', 'HIntReturns1',
       'HKickoffReturns0', 'HKickoffReturns1', 'HNetPassYards',
       'HPassesAttempted', 'HPassesCompleted', 'HPassesIntercepted',
       'HPenalties0', 'HPenalties1', 'HPossession', 'HPuntReturns0',
       'HPuntReturns1', 'HPunts0', 'HPunts1', 'HQB', 'HRushes0', 'HRushes1',
       'HSacked0', 'HSacked1', 'HomeScore', 'HomeTeam', 'Postseason',
       'PointsDiff'],
      dtype='object')

We can plot the point difference to get an idea of potentially suitable distributions:

.. code-block:: python

   data = pd.read_csv("nfl_data_new.csv")
   data["PointsDiff"] = data["HomeScore"] - data["AwayScore"]
   plt.figure(figsize=(15,7))
   plt.ylabel("Frequency")
   plt.xlabel("Points Difference")
   plt.hist(data["PointsDiff"],bins=20);

.. image:: http://www.pyflux.com/notebooks/GASRank/output_2_0.png

We will use a ``pf.Normal()`` families, although we could try a family with heavier tails also. We setup the :math:`GASRank` model, referring to the appropriate columns in our DataFrame:

.. code-block:: python

   model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                      score_diff="PointsDiff", family=pf.Normal())
   
Next we estimate the latent variables. For this example we will use a maximum likelihood point mass estimate :math:`z^{MLE}`: 

.. code-block:: python

   x = model.fit()
   x.summary()

   NormalGAS Rank                                                                                            
   ======================================== ==================================================
   Dependent Variable: PointsDiff           Method: MLE                                       
   Start Date: 0                            Log Likelihood: -10825.1703                       
   End Date: 2667                           AIC: 21656.3406                                   
   Number of observations: 2668             BIC: 21674.0079                                   
   ===========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== =========================
   Constant                  2.2405     0.2547     8.795    0.0      (1.7412 | 2.7398)        
   Ability Scale             0.0637     0.0058     10.9582  0.0      (0.0523 | 0.0751)        
   Normal Scale              13.9918                                                          
   ===========================================================================================

Once we have fit the model we can plot the power rankings of the teams in our DataFrame over their competitive history using :py:func:`plot_abilities`: 

.. code-block:: python

   model.plot_abilities(["Denver Broncos", "Green Bay Packers", "New England Patriots", 
                         "Carolina Panthers"],figsize=(15,8))

.. image:: http://www.pyflux.com/notebooks/GASRank/output_4_0.png

.. code-block:: python

   model.plot_abilities(["San Francisco 49ers", "Oakland Raiders", "San Diego Chargers"],
                          figsize=(15,8))

.. image:: http://www.pyflux.com/notebooks/GASRank/output_6_0.png

We can predict the point difference between two competitors in the future using :py:func:`predict`: 

.. code-block:: python

   model.predict("Denver Broncos","Carolina Panthers",neutral=True)
   array(-4.886816685966575)

Our DataFrame also has information on quarterbacks. Let's extend our model with a second component by including quarterbacks in the model:

.. code-block:: python

   model.add_second_component("HQB","AQB")
   x = model.fit()
   x.summary()

   NormalGAS Rank                                                                                            
   ======================================== ==================================================
   Dependent Variable: PointsDiff           Method: MLE                                       
   Start Date: 0                            Log Likelihood: -10799.4544                       
   End Date: 2667                           AIC: 21606.9087                                   
   Number of observations: 2668             BIC: 21630.4651                                   
   ===========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== =========================
   Constant                  2.2419     0.2516     8.9118   0.0      (1.7488 | 2.735)         
   Ability Scale 1           0.0186     0.0062     2.9904   0.0028   (0.0064 | 0.0307)        
   Ability Scale 2           0.0523     0.0076     6.8492   0.0      (0.0373 | 0.0673)        
   Normal Scale              13.8576                                                          
   ==========================================================================================================

We can plot the power rankings of the QBs in our DataFrame over their competitive history using :py:func:`plot_abilities`: 

.. code-block:: python

   model.plot_abilities(["Cam Newton", "Peyton Manning"],1,figsize=(15,8))

.. image:: http://www.pyflux.com/notebooks/GASRank/output_9_0.png

We can predict the point difference between two competitors in the future using :py:func:`predict`: 

.. code-block:: python

   model.predict("Denver Broncos","Carolina Panthers","Peyton Manning","Cam Newton",neutral=True)
   array(-7.33759714587138)

And some more power rankings for fan interest...

.. code-block:: python

   model.plot_abilities(["Aaron Rodgers", "Tom Brady", "Russell Wilson"],1,figsize=(15,8))

.. image:: http://www.pyflux.com/notebooks/GASRank/output_10_0.png

.. code-block:: python

   model.plot_abilities(["Peyton Manning","Michael Vick", "David Carr", "Carson Palmer"
                        ,"Eli Manning","Alex Smith","JaMarcus Russell","Matthew Stafford"
                        ,"Sam Bradford","Cam Newton","Andrew Luck","Jameis Winston"],1,
                        figsize=(15,8))

.. image:: http://www.pyflux.com/notebooks/GASRank/output_11_0.png

Class Description
----------

.. py:class:: GASRank(data, team_1, team_2, family, score_diff)

   **Generalized Autoregressive Score Ranking Models (GASRank).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.dataframe                       Containing the competitive data
   team_1               string                             Column name for home team names
   team_2               string                             Column name for away team names
   family               pf.Family instance                 The distribution for the time series,
                                                           e.g ``pf.Normal()``
   score_diff           string                             Column name for the point difference
   ==================   ===============================    ======================================

   **Attributes**

   .. py:attribute:: latent_variables

      A pf.LatentVariables() object containing information on the model latent variables, 
      prior settings. any fitted values, starting values, and other latent variable 
      information. When a model is fitted, this is where the latent variables are updated/stored. 
      Please see the documentation on Latent Variables for information on attributes within this
      object, as well as methods for accessing the latent variable information. 

   **Methods**

   .. py:method:: add_second_component(team_1, team_2)

      Adds a second component to the model

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      team_1               string                      Column name for team 1 second component
      team_2               string                      Column name for team 2 second component
      ==================   ========================    ======================================

      **Returns** : void - changes model to a second component model

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

   .. py:method:: plot_abilities(team_ids)
      
      Plots power rankings of the model components. Optional arguments include *figsize*,
      the dimensions of the figure to plot.

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      team_ids             list                        Of strings (team names) or indices
      ==================   ========================    ======================================

      For a two component model, arguments are:

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      team_ids             list                        Of strings (team names) or indices
      component_id         int                         0 for component 1, 1 for component 2
      ==================   ========================    ======================================
      
      **Returns** : void - shows a matplotlib plot

   .. py:method:: plot_fit(**kwargs)
      
      Plots the fit of the model against the data. Optional arguments include *figsize*,
      the dimensions of the figure to plot.

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

   .. py:method:: predict(team_1, team_2, neutral=False)
      
      Returns predicted point differences. For a one component model, arguments are:

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      team_1               string or int               If string, team name, else team index
      team_2               string or int               If string, team name, else team index
      neutral              boolean                     If True, disables home advantage
      ==================   ========================    ======================================

      For a two component model, arguments are:

      ==================   ========================    ======================================
      Parameter            Type                        Description
      ==================   ========================    ======================================
      team_1               string or int               If string, team name, else team index
      team_2               string or int               If string, team name, else team index
      team1b               string or int               If string, team 1, player 2 name
      team2b               string or int               If string, team 2, player 2 name
      neutral              boolean                     If True, disables home advantage
      ==================   ========================    ======================================
      
      **Returns** : np.ndarray - point difference predictions

References
----------

Creal, D; Koopman, S.J.; Lucas, A. (2013). Generalized Autoregressive Score Models with
Applications. Journal of Applied Econometrics, 28(5), 777–795. doi:10.1002/jae.1279.

Harvey, A.C. (2013). Dynamic Models for Volatility and Heavy Tails: With Applications to
Financial and Economic Time Series. Cambridge University Press.

Taylor, R. (2016). A Tour of Time Series Analysis (and a model for predicting NFL games). 
https://github.com/RJT1990/PyData2016-SanFrancisco