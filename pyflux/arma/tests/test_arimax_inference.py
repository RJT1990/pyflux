import numpy as np
import pandas as pd
from pyflux.arma import ARIMAX

# Set up some data to use for the tests

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


def test_bbvi():
    """
    Tests an ARIMAX model estimated with BBVI, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=200, record_elbo=True, quiet_progress=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=200, mini_batch=32, record_elbo=True, quiet_progress=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_mh():
    """
    Tests an ARIMAX model estimated with Metropolis-Hastings, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('M-H', nsims=200, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
    """
    Tests an ARIMAX model estimated with Laplace approximation, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
    """
    Tests an ARIMAX model estimated with PML, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1)
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)



def test_2_bbvi():
    """
    Tests an ARIMAX model estimated with BBVI, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_2_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_2_mh():
    """
    Tests an ARIMAX model estimated with MEtropolis-Hastings, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
    x = model.fit('M-H', nsims=200, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_2_laplace():
    """
    Tests an ARIMAX model estimated with Laplace, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_2_pml():
    """
    Tests an ARIMAX model estimated with PML, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1)
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

