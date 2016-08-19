import numpy as np
import pyflux as pf
import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime

a = DataReader('JPM',  'yahoo', datetime(2014,1,1), datetime(2016,3,10))
a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
a_returns.index = a.index.values[1:a.index.values.shape[0]]
a_returns.columns = ["JPM Returns"]

spy = DataReader('SPY',  'yahoo', datetime(2014,1,1), datetime(2016,3,10))
spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
spy_returns.index = spy.index.values[1:spy.index.values.shape[0]]
spy_returns.columns = ['S&P500 Returns']

one_mon = DataReader('DGS1MO', 'fred',datetime(2014,1,1), datetime(2016,3,10))
one_day = np.log(1+one_mon)/365

returns = pd.concat([one_day,a_returns,spy_returns],axis=1).dropna()
excess_m = returns["JPM Returns"].values - returns['DGS1MO'].values
excess_spy = returns["S&P500 Returns"].values - returns['DGS1MO'].values
data = pd.DataFrame(np.transpose([excess_m,excess_spy, returns['DGS1MO'].values]))
data.columns=["JPM","SP500","Rf"]
data.index = returns.index

def test_no_terms():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=0, q=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=1, q=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=1, q=1)
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=1, q=1)
	x = model.fit('M-H', nsims=300)
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=1, q=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=1, q=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5, oos_data=data).shape[0] == 5)

def test_predict_is_length():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=2, q=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5, oos_data=data).values[np.isnan(model.predict(h=5, oos_data=data).values)]) == 0)

def test_predict_is_nans():
	model = pf.EGARCHMReg(formula='JPM~Rf', data=data, q=2, p=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)
