import numpy as np
import pyflux as pf

noise = np.random.normal(0,1,200)
data = np.zeros(200)

for i in range(1,len(data)):
    data[i] = 1.0*data[i-1] + noise[i]

countdata = np.random.poisson(3,200)

def test_skewt_couple_terms():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_skewt_couple_terms_integ():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASLLT(data=data, integ=1, family=pf.Skewt())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_skewt_bbvi():
    """
    Tests an GAS model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    """

def test_skewt_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    """
    
def test_skewt_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('BBVI',iterations=100, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_skewt_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('BBVI',iterations=100, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_skewt_mh():
    """
    Tests an GAS model estimated with Metropolis-Hastings and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    """

""" Uncomment in future if Skewt becomes more robust
def test_skewt_laplace():
    Tests an GAS model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
"""

def test_skewt_pml():
    """
    Tests a PML model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_skewt_predict_length():
    """
    Tests that the prediction dataframe length is equal to the number of steps h
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5).shape[0] == 5)

def test_skewt_predict_is_length():
    """
    Tests that the prediction IS dataframe length is equal to the number of steps h
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_skewt_predict_nans():
    """
    Tests that the predictions are not nans
    model = pf.GASLLT(data=data, family=pf.Skewt())
    """
    """
    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)
    """
"""

def test_skewt_predict_is_nans():

    Tests that the in-sample predictions are not nans

    model = pf.GASLLT(data=data, family=pf.Skewt())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)
"""

