import numpy as np
import pyflux as pf
import pandas as pd

noise = np.random.normal(0,0.000001,1000)
x1 = np.random.normal(0,0.00001,1000)
x2 = np.random.normal(0,0.00001,1000)

vol = np.ones(1000)*0.02

for i in range(1,len(vol)):
	vol[i] = 0.999*vol[i-1] + noise[i]

y = np.random.normal(0,vol,100)

data = pd.DataFrame([y,x1,x2]).T
data.columns = ['y', 'x1', 'x2']

y_oos = np.random.normal(0,1,30)
x1_oos = np.random.normal(0,1,30)
x2_oos = np.random.normal(0,1,30)

data_oos = pd.DataFrame([y_oos,x1_oos,x2_oos]).T
data_oos.columns = ['y', 'x1', 'x2']

def test_no_terms():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=0, q=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 8)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 10)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 10)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

#def test_mh():
#	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
#	x = model.fit('M-H', nsims=300)
#	assert(len(model.latent_variables.z_list) == 10)
#	lvs = np.array([i.value for i in model.latent_variables.z_list])
#	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 10)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 10)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test_predict_is_length():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, oos_data=data_oos).values)]) == 0)

def test_predict_is_nans():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, q=2, p=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)



def test_lev_no_terms():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=0, q=0)
	model.add_leverage()
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 9)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_couple_terms():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 11)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_bbvi():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('BBVI', iterations=100)
	assert(len(model.latent_variables.z_list) == 11)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

#def test_lev_mh():
#	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
#	model.add_leverage()
#	x = model.fit('M-H', nsims=300)
#	assert(len(model.latent_variables.z_list) == 11)
#	lvs = np.array([i.value for i in model.latent_variables.z_list])
#	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_laplace():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 11)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_pml():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=1, q=1)
	model.add_leverage()
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 11)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_predict_length():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test_lev_predict_is_length():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_lev_predict_nans():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, p=2, q=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, oos_data=data_oos).values)]) == 0)

def test_lev_predict_is_nans():
	model = pf.EGARCHMReg(formula='y~x1+x2', data=data, q=2, p=2)
	model.add_leverage()
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)