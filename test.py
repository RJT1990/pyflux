import numpy as np

print np.random.normal(0, 1, 1)[0]

import pyflux as pf
from math import sqrt
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as ss
import timeit
import pandas as pd
ibm = DataReader('IBM',  'yahoo', datetime(2000,1,1), datetime(2016,3,10))
x = ibm['Open'].values
z = ibm.index.dayofweek
ibm['Logged Open'] = np.log(ibm['Open'].values)


y = pf.GAS(ar=1,sc=1,integ=1,data=ibm,dist="Laplace")
y.fit("MLE")


"""
y = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1,target='Logged Open')
y.fit("MLE")

y.list_priors()
y.adjust_prior(1,pf.Normal(0,0.3,None))
y.list_priors()

pf.acf_plot(np.diff(x),max_lag=10)

#y.fit(method="M-H",cov_matrix=np.array([[1,0,0,0],[0,1,-0.5,0],[0,-0.5,1,0],[0,0,0,1]]))

ibm = DataReader('IBM',  'yahoo', datetime(2000,1,1), datetime(2016,3,10))
x = ibm['Open'].values
z = ibm.index.dayofweek

y = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1)
y.fit("MLE")

y = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1)


y = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1)
print y.param_desc
y = pf.GAS(ar=1,sc=1,integ=1,data=ibm,dist="Laplace")

y = pf.ARIMA(data=ibm,ar=1,ma=1,integ=1)
y = pf.ARIMA(data=ibm,ar=2,ma=1,integ=1)
y = pf.GAS(ar=1,sc=1,integ=1,data=ibm,dist="Laplace")
y.fit("MAP")

y = pf.ARIMA(ar=2,ma=1,integ=1,data=ibm)
y.fit("Laplace")
print y.predict(T=5,lookback=10)
print y.chains[0]
print np.random.random(y.chains[0],1)
y = pf.GAS(ar=1,sc=1,integ=1,data=ibm,dist="Laplace")
y.fit(method="M-H")
"""




"""
print y.params

"""
"""
y = pf.ARIMA(ar=2,ma=1,integ=1,data=x)
y.fit(method="MAP")
y.fit(method="M-H")


predictions = y.predict(T=50)
plt.plot(predictions[len(predictions)-50:len(predictions)-1])
plt.show()
scale = 2.38/30
plt.plot(x)
plt.show()
y.mh_fit(nsims=10000)
"""