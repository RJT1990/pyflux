import numpy as np
from pyflux.arma import ARIMA

noise = np.random.normal(0,1,100)
data = np.zeros(100)

for i in range(1,len(data)):
    data[i] = 0.9*data[i-1] + noise[i]


def test_bbvi():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('BBVI',iterations=100, quiet_progress=True, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_mh():
    """
    Tests an ARIMA model estimated with Metropolis-Hastings and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('M-H', nsims=200, quiet_progress=True)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
    """
    Tests an ARIMA model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
    """
    Tests a PML model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
