Getting Started with Time Series
==================================

Introduction
----------

Time series analysis is a subfield of statistics and econometrics. Time series data :math:`y_{t}` is indexed by time :math:`t` and ordered sequentially. This presents unique challenges including autocorrelation within the data, non-exchangeability of data points, and non-stationarity of data and parameters. Because of the sequential nature of the data, time series analysis has particular goals. We can summarize these goals into one of **description**  of a time series in terms of latent components or features of interest, and **prediction**, which aims to produce reasonable forecasts of the future (Harvey, 1990). 

From start to finish, we can place time series modelling in a framework in the spirit of **Box's Loop** (Blei, D.M. 2014). In particular, we:

#. Build a model for the time series data
#. Perform inference on the model
#. Check the model fit, performing evaluation & criticism
#. Revise the model, repeat until happy
#. Perform retrospection and prediction with the model

Below we outline an example model building process for JPMorgan Chase stock data, where the index is daily. Consider this time series data:

::

  import pandas as pd
  import numpy as np
  from pandas_datareader.data import DataReader
  from datetime import datetime

  a = DataReader('JPM',  'yahoo', datetime(2006,6,1), datetime(2016,6,1))
  a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
  a_returns.index = a.index.values[1:a.index.values.shape[0]]
  a_returns.columns = ["JPM Returns"]

  a_returns.head()

.. raw:: html

  <div>
  <table border="1" class="dataframe", background='#FFFFFF'>
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>JPM Returns</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>2006-06-02</th>
        <td>0.005264</td>
      </tr>
      <tr>
        <th>2006-06-05</th>
        <td>-0.019360</td>
      </tr>
      <tr>
        <th>2006-06-06</th>
        <td>-0.013826</td>
      </tr>
      <tr>
        <th>2006-06-07</th>
        <td>-0.003072</td>
      </tr>
      <tr>
        <th>2006-06-08</th>
        <td>0.002364</td>
      </tr>
    </tbody>
  </table>
  </div>
  <br>

The index of the data is meaningful for this data; we cannot simply 'shuffle the deck' otherwise we could lose meaningful dependencies such as seasonality, trends, cycles and other components.

Step One: Visualize the Data
----------

Because time series is sequential, plotting the data allows us to obtain an idea of its properties. We can also plot autocorrelation plots of the data (and transformations of the data) to understand if autocorrelation exists in the series. Lastly, in this stage, we can reason about potential features that might explain variation in the series.


For our stock market data, we can first plot the data:

::

  plt.figure(figsize=(15, 5))
  plt.ylabel("Returns")
  plt.plot(a_returns)
  plt.show()

.. image:: http://www.pyflux.com/welcome_pack/introduction/output_2_0.png

It appears that the volatility of the series changes over time, and is clustering in periods of market turbulence, such as in the financial crisis of 2008. We can obtain more insight by plotting autocorrelation functions of the returns and squared returns:


::

  import pyflux as pf
  import matplotlib.pyplot as plt
  pf.acf_plot(a_returns.values.T[0])
  pf.acf_plot(np.square(a_returns.values.T[0]))

.. image:: http://www.pyflux.com/welcome_pack/introduction/output_3_0.png

.. image:: http://www.pyflux.com/welcome_pack/introduction/output_3_1.png

The squared returns demonstrate strong evidence of autocorrelation. The fact that autocorrelation persists and decays over multiply lags is evidence of an autoregressive effect within volatility. For returns, there is less strong evidence of autocorrelation, although the first lag is significant.

Step Two: Propose a Model
----------

We reason about a model that can explain the variation in the data and we specify any prior beliefs we have about the model parameters. We saw evidence of volatility clustering. One way to model this effect is through a GARCH model for volatility (Bollerslev, T. 1986).

.. math::
  
  y_{t} \sim N\left(\mu,\sigma_{t}\right)
.. math::

  \sigma_{t}^{2} = \omega + \alpha\epsilon_{t}^{2} + \beta{\sigma_{t-1}^{2}}

We will perform Bayesian inference on this model, and so we will specify some priors. We will ensure :math:`\omega > 0` through a log transform, and we will use a Truncated Normal prior on :math:`\alpha, \beta`:

::

  my_model = pf.GARCH(p=1, q=1, data=a_returns)
  print(my_model.latent_variables)

    Index    Latent Variable     Prior           Prior Hyperparameters   V.I. Dist  Transform 
    ======== =================== =============== ======================= ========== ==========
    0        Vol Constant        Normal          mu0: 0, sigma0: 3       Normal     exp       
    1        q(1)                Normal          mu0: 0, sigma0: 0.5     Normal     logit     
    2        p(1)                Normal          mu0: 0, sigma0: 0.5     Normal     logit     
    3        Returns Constant    Normal          mu0: 0, sigma0: 3       Normal     None      

  my_model.adjust_prior(1, pf.TruncatedNormal(0.01, 0.5, lower=0.0, upper=1.0))
  my_model.adjust_prior(2, pf.TruncatedNormal(0.97, 0.5, lower=0.0, upper=1.0))


Step Three: Perform Inference
----------

As a third step we need to decide how to perform inference for the model. Below we use 
Metropolis-Hastings for approximate inference on our GARCH model. We also plot the latent variables :math:`\alpha` and :math:`\beta`:

::

  result = my_model.fit('M-H', nsims=20000)

  Tuning complete! Now sampling.
  Acceptance rate of Metropolis-Hastings is 0.33865

  my_model.plot_z([1,2])

.. image:: http://www.pyflux.com/welcome_pack/introduction/output_7_0.png

Step Four: Evaluate Model Fit
----------

We next evaluate the fit of the model and establish whether we can improve the model further. For time series, the simplest way to visualize fit is to plot the series against its predicted values; we can also check out-of-sample performance. If we seek further model improvements, we go back to **step two** and proceed. Once we are happy we go to **step five**.

Below we plot the fit of the GARCH model and observe that it picking up volatility clustering in the series:

::

  my_model.plot_fit(figsize=(15,5))


.. image:: http://www.pyflux.com/welcome_pack/introduction/output_8_0.png

We can also plot samples from the posterior predictive density:

::
  
  my_model.plot_sample(nsims=10, figsize=(15,7))

.. image:: http://www.pyflux.com/welcome_pack/introduction/plot_sample.png

We can see that the samples (colored) appear to be picking up variation in the data (the square datapoints). 

We can also perform a posterior predictive check (PPC) on features of the generated series, for example the kurtosis:

::

  from scipy.stats import kurtosis
  my_model.plot_ppc(T=kurtosis)

.. image:: http://www.pyflux.com/welcome_pack/introduction/plot_ppc.png

It appears our generated data underestimates kurtosis in the series. This is not surprising as we are assuming normally distributed returns, so we may want to consider alternative volatility models.

Step Five: Analyse and Predict
----------

Once we are happy with our model, we can use it to analyze the historical time series and make predictions. For our GARCH model, we can see from the previous fit plot that the main periods of volatility picked up are during the financial crisis of 2007-2008, and during the Eurozone crisis in late 2011. We can also obtain forward predictions with the model:

::
  
  my_model.plot_predict(h=30, figsize=(15,5))

.. image:: http://www.pyflux.com/welcome_pack/introduction/plot_predict.png

References
----------

Blei, D. M. (2014). Build, compute, critique, repeat: Data analysis with latent variable models. Annual Review of Statistics and Its Application, 1, 203–232.

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal
of Econometrics. April, 31:3, pp. 307–27.

Harvey A. C. (1990). Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press.

