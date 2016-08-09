import numpy as np
import pyflux as pf

noise = np.random.normal(0,0.00001,100)
vol = np.ones(100)*0.01

for i in range(1,len(vol)):
	vol[i] = 0.999*vol[i-1] + noise[i]

data = np.random.normal(0,vol,100)

def test_no_terms():
	model = pf.GARCH(data=data, p=0, q=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 2)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
	model = pf.GARCH(data=data, p=1, q=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	model = pf.GARCH(data=data, p=1, q=1)
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
	model = pf.GARCH(data=data, p=1, q=1)
	x = model.fit('M-H', nsims=300)
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
	model = pf.GARCH(data=data, p=1, q=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
	model = pf.GARCH(data=data, p=1, q=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	model = pf.GARCH(data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
	model = pf.GARCH(data=data, p=2, q=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	model = pf.GARCH(data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
	model = pf.GARCH(data=data, q=2, p=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)