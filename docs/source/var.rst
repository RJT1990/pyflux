VAR models
==================================

Introduction
----------

Vector autoregressions (VARs) were introduced in the econometrics literature in the 1980s to allow for (linear) dependencies among multiple variables. For a :math:`K` x :math:`1` vector :math:`y_{t}` we can specify a VAR(p) model as:

.. math::

   y_{t} = c + A_{1}y_{t-1} + ... + A_{p}y_{t-p} + e_{t}

These models can be estimated quickly through OLS. But with a large number of dependent variables, the number of parameters to be estimated can grow very quickly. See the notebook on **Bayesian VARs** for an alternative way to approach these types of model.

Example
----------

We'll run an VAR model for US banking sector stocks.

.. code-block:: python

   import numpy as np
   import pyflux as pf
   from pandas_datareader import DataReader
   from datetime import datetime
   import matplotlib.pyplot as plt
   %matplotlib inline 

   ibm = DataReader(['JPM','GS','BAC','C','WFC','MS'],  'yahoo', datetime(2012,1,1), datetime(2016,6,28))
   opening_prices = np.log(ibm['Open'])
   plt.figure(figsize=(15,5));
   plt.plot(opening_prices.index,opening_prices);
   plt.legend(opening_prices.columns.values,loc=3);
   plt.title("Logged opening price");

.. image:: http://www.pyflux.com/notebooks/VAR/output_7_1.png

Here we specify an arbitrary VAR(2) model, which we fit via OLS: 

.. code-block:: python

   model = pf.VAR(data=opening_prices, lags=2, integ=1)
 
Next we estimate the latent variables. For this example we will use an OLS estimate :math:`z^{OLS}`: 

.. code-block:: python

   x = model.fit()
   x.summary()

   VAR(2)                                                                                                    
   ======================================== =================================================
   Dependent Variable: Differenced BAC      Method: OLS                                       
   Start Date: 2012-01-05 00:00:00          Log Likelihood: 21547.2578                        
   End Date: 2016-06-28 00:00:00            AIC: -42896.5156                                  
   Number of observations: 1126             BIC: -42398.8994                                  
   ==========================================================================================
   Latent Variable           Estimate   Std Error  z        P>|z|    95% C.I.                 
   ========================= ========== ========== ======== ======== ========================
   Diff BAC Constant         0.0007     0.0006     1.2007   0.2299   (-0.0004 | 0.0018)       
   Diff BAC AR(1)            -0.0525    0.0005     -97.5672 0.0      (-0.0535 | -0.0514)      
   Diff C to Diff BAC AR(1)  -0.0616    0.0004     -143.365 0.0      (-0.0625 | -0.0608)      
   Diff GS to Diff BAC AR(1) 0.0595     0.0004     132.6638 0.0      (0.0587 | 0.0604)        
   Diff JPM to Diff BAC AR(1)0.0296     0.0006     49.3563  0.0      (0.0284 | 0.0308)        
   Diff MS to Diff BAC AR(1) -0.0231    0.0004     -62.6218 0.0      (-0.0239 | -0.0224)      
   Diff WFC to Diff BAC AR(1)-0.0417    0.0598     -0.6968  0.4859   (-0.159 | 0.0756)        
   Diff BAC AR(2)            0.1171     0.0555     2.1087   0.035    (0.0083 | 0.226)         
   Diff C to Diff BAC AR(2)  -0.1266    0.0444     -2.8528  0.0043   (-0.2136 | -0.0396)      
   Diff GS to Diff BAC AR(2) 0.1698     0.0464     3.6618   0.0003   (0.0789 | 0.2606)        
   Diff JPM to Diff BAC AR(2)-0.0959    0.062      -1.5472  0.1218   (-0.2174 | 0.0256)       
   Diff MS to Diff BAC AR(2) -0.0213    0.0381     -0.557   0.5775   (-0.096 | 0.0535)        
   Diff WFC to Diff BAC AR(2)-0.001     0.0701     -0.0149  0.9881   (-0.1384 | 0.1363)       
   Diff C Constant           0.0003     0.065      0.0047   0.9962   (-0.1272 | 0.1278)       
   Diff C AR(1)              0.0193     0.052      0.3706   0.7109   (-0.0826 | 0.1211)       
   Diff BAC to Diff C AR(1)  -0.0576    0.0543     -1.0613  0.2886   (-0.164 | 0.0488)        
   Diff GS to Diff C AR(1)   0.0579     0.0726     0.7979   0.4249   (-0.0844 | 0.2002)       
   Diff JPM to Diff C AR(1)  0.0831     0.0447     1.8595   0.063    (-0.0045 | 0.1706)       
   Diff MS to Diff C AR(1)   -0.037     0.0794     -0.4657  0.6414   (-0.1925 | 0.1186)       
   Diff WFC to Diff C AR(1)  -0.1785    0.0737     -2.4235  0.0154   (-0.3229 | -0.0341)      
   Diff C AR(2)              0.1612     0.0589     2.7379   0.0062   (0.0458 | 0.2765)        
   Diff BAC to Diff C AR(2)  -0.1021    0.0615     -1.6598  0.0969   (-0.2226 | 0.0185)       
   Diff GS to Diff C AR(2)   0.1109     0.0822     1.3483   0.1776   (-0.0503 | 0.272)        
   Diff JPM to Diff C AR(2)  -0.0453    0.0506     -0.8946  0.371    (-0.1444 | 0.0539)       
   Diff MS to Diff C AR(2)   0.0127     0.0775     0.1643   0.8695   (-0.1391 | 0.1646)       
   Diff WFC to Diff C AR(2)  -0.1313    0.0719     -1.8261  0.0678   (-0.2723 | 0.0096)       
   Diff GS Constant          0.0003     0.0575     0.006    0.9952   (-0.1123 | 0.113)        
   Diff GS AR(1)             -0.016     0.06       -0.266   0.7903   (-0.1336 | 0.1017)       
   Diff BAC to Diff GS AR(1) 0.0051     0.0803     0.0633   0.9495   (-0.1523 | 0.1624)       
   Diff C to Diff GS AR(1)   -0.0785    0.0494     -1.5891  0.112    (-0.1753 | 0.0183)       
   Diff JPM to Diff GS AR(1) 0.0507     0.0575     0.8814   0.3781   (-0.062 | 0.1633)        
   Diff MS to Diff GS AR(1)  0.0425     0.0534     0.7961   0.4259   (-0.0621 | 0.1471)       
   Diff WFC to Diff GS AR(1) -0.0613    0.0426     -1.4376  0.1505   (-0.1449 | 0.0223)       
   Diff GS AR(2)             0.0865     0.0445     1.9422   0.0521   (-0.0008 | 0.1738)       
   Diff BAC to Diff GS AR(2) -0.1896    0.0596     -3.1832  0.0015   (-0.3064 | -0.0729)      
   Diff C to Diff GS AR(2)   0.0423     0.0367     1.1553   0.248    (-0.0295 | 0.1142)       
   Diff JPM to Diff GS AR(2) 0.0667     0.0769     0.8664   0.3863   (-0.0841 | 0.2174)       
   Diff MS to Diff GS AR(2)  0.0433     0.0714     0.6067   0.5441   (-0.0966 | 0.1833)       
   Diff WFC to Diff GS AR(2) -0.0362    0.0571     -0.6347  0.5256   (-0.1481 | 0.0756)       
   Diff JPM Constant         0.0005     0.0596     0.0082   0.9934   (-0.1163 | 0.1173)       
   Diff JPM AR(1)            -0.0304    0.0797     -0.3813  0.703    (-0.1866 | 0.1258)       
   Diff BAC to Diff JPM AR(1)-0.0281    0.049      -0.5738  0.5661   (-0.1243 | 0.068)        
   Diff C to Diff JPM AR(1)  0.0695     0.0594     1.1698   0.2421   (-0.047 | 0.186)         
   Diff GS to Diff JPM AR(1) -0.0106    0.0552     -0.1924  0.8474   (-0.1187 | 0.0975)       
   Diff MS to Diff JPM AR(1) -0.0338    0.0441     -0.7675  0.4428   (-0.1202 | 0.0526)       
   Diff WFC to Diff JPM AR(1)-0.0725    0.046      -1.5744  0.1154   (-0.1627 | 0.0178)       
   Diff JPM AR(2)            0.096      0.0616     1.559    0.119    (-0.0247 | 0.2167)       
   Diff BAC to Diff JPM AR(2)-0.1246    0.0379     -3.2883  0.001    (-0.1989 | -0.0503)      
   Diff C to Diff JPM AR(2)  0.0229     0.0696     0.3284   0.7426   (-0.1136 | 0.1593)       
   Diff GS to Diff JPM AR(2) -0.0084    0.0646     -0.1301  0.8965   (-0.1351 | 0.1182)       
   Diff MS to Diff JPM AR(2) 0.0319     0.0516     0.6182   0.5364   (-0.0693 | 0.1332)       
   Diff WFC to Diff JPM AR(2)-0.0117    0.0539     -0.2161  0.8289   (-0.1174 | 0.0941)       
   Diff MS Constant          0.0004     0.0721     0.005    0.996    (-0.141 | 0.1417)        
   Diff MS AR(1)             0.0249     0.0444     0.5605   0.5752   (-0.0621 | 0.1119)       
   Diff BAC to Diff MS AR(1) 0.0456     0.0783     0.5833   0.5597   (-0.1077 | 0.199)        
   Diff C to Diff MS AR(1)   0.0083     0.0726     0.1148   0.9086   (-0.134 | 0.1507)        
   Diff GS to Diff MS AR(1)  0.1319     0.0581     2.2717   0.0231   (0.0181 | 0.2457)        
   Diff JPM to Diff MS AR(1) -0.1771    0.0606     -2.9213  0.0035   (-0.296 | -0.0583)       
   Diff WFC to Diff MS AR(1) -0.151     0.0811     -1.8629  0.0625   (-0.31 | 0.0079)         
   Diff MS AR(2)             0.1512     0.0499     3.0308   0.0024   (0.0534 | 0.249)         
   Diff BAC to Diff MS AR(2) -0.2173    0.0772     -2.8157  0.0049   (-0.3686 | -0.066)       
   Diff C to Diff MS AR(2)   0.1827     0.0716     2.5499   0.0108   (0.0423 | 0.3231)        
   Diff GS to Diff MS AR(2)  -0.0107    0.0573     -0.1873  0.8514   (-0.1229 | 0.1015)       
   Diff JPM to Diff MS AR(2) 0.0004     0.0598     0.0066   0.9947   (-0.1168 | 0.1176)       
   Diff WFC to Diff MS AR(2) -0.0697    0.08       -0.8711  0.3837   (-0.2264 | 0.0871)       
   Diff WFC Constant         0.0005     0.0492     0.0095   0.9924   (-0.096 | 0.0969)        
   Diff WFC AR(1)            0.0092     0.0574     0.1611   0.872    (-0.1032 | 0.1217)       
   Diff BAC to Diff WFC AR(1)-0.0059    0.0532     -0.1113  0.9114   (-0.1103 | 0.0984)       
   Diff C to Diff WFC AR(1)  0.0062     0.0425     0.1448   0.8848   (-0.0772 | 0.0896)       
   Diff GS to Diff WFC AR(1) 0.0525     0.0444     1.1811   0.2376   (-0.0346 | 0.1396)       
   Diff JPM to Diff WFC AR(1)-0.0047    0.0594     -0.0792  0.9368   (-0.1212 | 0.1118)       
   Diff MS to Diff WFC AR(1) -0.1996    0.0366     -5.4578  0.0      (-0.2713 | -0.1279)      
   Diff WFC AR(2)            0.0291     0.0773     0.3759   0.707    (-0.1225 | 0.1806)       
   Diff BAC to Diff WFC AR(2)-0.0509    0.0718     -0.7087  0.4785   (-0.1915 | 0.0898)       
   Diff C to Diff WFC AR(2)  0.0255     0.0574     0.4444   0.6567   (-0.0869 | 0.1379)       
   Diff GS to Diff WFC AR(2) 0.0235     0.0599     0.3922   0.6949   (-0.0939 | 0.1409)       
   Diff JPM to Diff WFC AR(2)0.015      0.0801     0.1878   0.851    (-0.142 | 0.1721)        
   Diff MS to Diff WFC AR(2) -0.0556    0.0493     -1.1276  0.2595   (-0.1522 | 0.041)        
   ==========================================================================================

We can plot latent variables with :py:func:`plot_z`: method:

.. code-block:: python

   model.plot_z(list(range(0,6)),figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/VAR/output_11_0.png

We can plot the in-sample fit with :py:func:`plot_fit`:

.. code-block:: python

   model.plot_fit(figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/VAR/output_13_0.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_13_1.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_13_2.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_13_3.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_13_4.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_13_5.png

We can make forward predictions with our model using :py:func:`plot_predict`:

.. code-block:: python

   model.plot_predict(past_values=19, h=5, figsize=(15,5))

.. image:: http://www.pyflux.com/notebooks/VAR/output_15_0.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_15_1.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_15_2.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_15_3.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_15_4.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_15_5.png

How does our model perform? We can get a sense by performing a rolling in-sample prediction – :py:func:`plot_predict_is`: for plotted graphs: 

.. code-block:: python

   model.plot_predict_is(h=30, figsize=((15,5)))

.. image:: http://www.pyflux.com/notebooks/VAR/output_19_0.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_19_1.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_19_2.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_19_3.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_19_4.png
.. image:: http://www.pyflux.com/notebooks/VAR/output_19_5.png


Class Description
----------

.. py:class:: VAR(data, lags, integ, target, use_ols_covariance)

   **Vector Autoregression Models (VAR).**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   data                 pd.DataFrame or np.ndarray         Contains the univariate time series
   lags                 int                                The number of autoregressive lags
   integ                int                                How many times to difference the data
                                                           (default: 0)
   target               string or int                      Which column of DataFrame/array to use.
   use_ols_covariance   boolean                            Whether to use fixed OLS covariance
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

Lütkepohl, H. & Kraetzig, M. (2004). Applied Time Series Econometrics. Cambridge University Press, Cambridge.
