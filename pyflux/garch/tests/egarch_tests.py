import numpy as np
import pyflux as pf

noise = np.random.normal(0,0.00001,100)
vol = np.ones(100)*0.01

for i in range(1,len(vol)):
	vol[i] = 0.999*vol[i-1] + noise[i]

data = np.random.normal(0,vol,100)

def test_no_terms():
	model = pf.EGARCH(data=data, p=0, q=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 3)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
	model = pf.EGARCH(data=data, p=1, q=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	model = pf.EGARCH(data=data, p=1, q=1)
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
	model = pf.EGARCH(data=data, p=1, q=1)
	x = model.fit('M-H', nsims=300)
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
	model = pf.EGARCH(data=data, p=1, q=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
	model = pf.EGARCH(data=data, p=1, q=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	model = pf.EGARCH(data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
	model = pf.EGARCH(data=data, p=2, q=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	model = pf.EGARCH(data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
	model = pf.EGARCH(data=data, q=2, p=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)



def test_lev_no_terms():
	model = pf.EGARCH(data=data, p=0, q=0)
	model.add_leverage()
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_couple_terms():
	model = pf.EGARCH(data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_bbvi():
	model = pf.EGARCH(data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_mh():
	model = pf.EGARCH(data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('M-H', nsims=300)
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_laplace():
	model = pf.EGARCH(data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_pml():
	model = pf.EGARCH(data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_predict_length():
	model = pf.EGARCH(data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(model.predict(h=5).shape[0] == 5)

def test_lev_predict_is_length():
	model = pf.EGARCH(data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_lev_predict_nans():
	model = pf.EGARCH(data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_lev_predict_is_nans():
	model = pf.EGARCH(data=data, q=2, p=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)