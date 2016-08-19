import numpy as np
import pyflux as pf
import pandas as pd

noise_1 = np.random.normal(0,1,350)
noise_2 = np.random.normal(0,1,350)
data_1 = np.zeros(350)
data_2 = np.zeros(350)

for i in range(1,len(data_1)):
	data_1[i] = 0.9*data_1[i-1] + noise_1[i]
	data_2[i] = 0.9*data_2[i-1] + noise_2[i]

data = pd.DataFrame([data_1,data_2]).T
data.columns = ['test1','test2']

# Uncomment once PML/Laplace approximation is more robust

def test_couple_terms():
	"""
	Tests an VAR model with 1 AR and 1 MA term and that
	the latent variable list length is correct, and that the estimated
	latent variables are not nan
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 13)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
	"""
	Tests an VAR model with 1 AR and 1 MA term, integrated once, and that
	the latent variable list length is correct, and that the estimated
	latent variables are not nan
	"""
	model = pf.VAR(data=data, lags=2, integ=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 13)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	"""
	Tests an VAR model estimated with BBVI and that the length of the latent variable
	list is correct, and that the estimated latent variables are not nan
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit('BBVI',iterations=100)
	assert(len(model.latent_variables.z_list) == 13)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
	"""
	Tests an VAR model estimated with Metropolis-Hastings and that the length of the 
	latent variable list is correct, and that the estimated latent variables are not nan
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit('M-H',nsims=300)
	assert(len(model.latent_variables.z_list) == 13)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

#def test_laplace():
#	"""
#	Tests an VAR model estimated with Laplace approximation and that the length of the 
#	latent variable list is correct, and that the estimated latent variables are not nan
#	"""
#	model = pf.VAR(data=data, lags=2)
#	x = model.fit('Laplace')
#	assert(len(model.latent_variables.z_list) == 13)
#	lvs = np.array([i.value for i in model.latent_variables.z_list])
#	assert(len(lvs[np.isnan(lvs)]) == 0)

#def test_pml():
#	"""
#	Tests a PML model estimated with Laplace approximation and that the length of the 
#	latent variable list is correct, and that the estimated latent variables are not nan
#	"""
#	model = pf.VAR(data=data, lags=2)
#	x = model.fit('PML')
#	assert(len(model.latent_variables.z_list) == 13)
#	lvs = np.array([i.value for i in model.latent_variables.z_list])
#	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	"""
	Tests that the prediction dataframe length is equal to the number of steps h
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit()
	assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
	"""
	Tests that the prediction IS dataframe length is equal to the number of steps h
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	"""
	Tests that the predictions are not nans
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit()
	assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
	"""
	Tests that the in-sample predictions are not nans
	"""
	model = pf.VAR(data=data, lags=2)
	x = model.fit()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)