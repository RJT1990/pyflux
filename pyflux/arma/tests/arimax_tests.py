import numpy as np
import pandas as pd
import pyflux as pf

noise = np.random.normal(0,1,100)
y = np.zeros(100)
x1 = np.random.normal(0,1,100)
x2 = np.random.normal(0,1,100)


for i in range(1,len(y)):
	y[i] = 0.9*y[i-1] + noise[i] + 0.1*x1[i] - 0.3*x2[i]

data = pd.DataFrame([y,x1,x2]).T
data.columns = ['y', 'x1', 'x2']

y_oos = np.random.normal(0,1,30)
x1_oos = np.random.normal(0,1,30)
x2_oos = np.random.normal(0,1,30)

data_oos = pd.DataFrame([y_oos,x1_oos,x2_oos]).T
data_oos.columns = ['y', 'x1', 'x2']

def test_no_terms():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=0, ma=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 3)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, integ=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
	x = model.fit('BBVI',iterations=100)
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
	x = model.fit('M-H',nsims=300)
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 5)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test_predict_is_length():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
		oos_data=data_oos).values)]) == 0)

def test_predict_is_nans():
	model = pf.ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

## Try more than one predictor

def test2_no_terms():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=0, ma=0)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 4)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_couple_terms():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
	x = model.fit()
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_bbvi():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
	x = model.fit('BBVI',iterations=100)
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_mh():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
	x = model.fit('M-H',nsims=300)
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_laplace():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
	x = model.fit('Laplace')
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_pml():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
	x = model.fit('PML')
	assert(len(model.latent_variables.z_list) == 6)
	lvs = np.array([i.value for i in model.latent_variables.z_list])
	assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_predict_length():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test2_predict_is_length():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2)
	x = model.fit()
	assert(model.predict_is(h=5).shape[0] == 5)

def test2_predict_nans():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
		oos_data=data_oos).values)]) == 0)

def test2_predict_is_nans():
	model = pf.ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2)
	x = model.fit()
	x.summary()
	assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)