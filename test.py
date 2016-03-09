import pyflux as pf
import numpy as np
from math import sqrt
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt

ibm = DataReader('IBM',  'yahoo', datetime(2000,1,1), datetime(2016,1,2))
x = ibm['Open'].values
z = ibm.index.dayofweek

y = pf.GAS(ar=1,sc=1,integ=1,data=x,dist="Laplace")
y.fit()
print y.params
"""
y = pf.ARI(ar=2,integ=1,data=x)
y.fit()
predictions = y.predict(T=50)
plt.plot(predictions[len(predictions)-50:len(predictions)-1])
plt.show()
scale = 2.38/30
plt.plot(x)
plt.show()
y.mh_fit(nsims=10000)
"""