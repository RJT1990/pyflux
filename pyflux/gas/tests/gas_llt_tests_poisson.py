import numpy as np
import pyflux as pf
import pandas as pd

data = pd.read_csv('http://www.pyflux.com/notebooks/eastmidlandsderby.csv')
total_goals = pd.DataFrame(data['Forest'] + data['Derby'])
data = total_goals.values.T[0]

def test_poisson_couple_terms():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_poisson_bbvi():
    """
    Tests an GAS model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    """

def test_poisson_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI',iterations=300, mini_batch=32, map_start=False)
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    """
def test_poisson_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI',iterations=300, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])
    """

def test_poisson_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI',iterations=200, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])
    """

def test_poisson_mh():
    """
    Tests an GAS model estimated with Metropolis-Hastings and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('PML')
    x = model.fit('M-H', nsims=300)
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_poisson_laplace():
    """
    Tests an GAS model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_poisson_pml():
    """
    Tests a PML model estimated with Laplace approximation and that the length of the 
    latent variable list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_poisson_predict_length():
    """
    Tests that the prediction dataframe length is equal to the number of steps h
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5).shape[0] == 5)

def test_poisson_predict_is_length():
    """
    Tests that the prediction IS dataframe length is equal to the number of steps h
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_poisson_predict_nans():
    """
    Tests that the predictions are not nans
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_poisson_predict_is_nans():
    """
    Tests that the in-sample predictions are not nans
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)
    
def test_poisson_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def test_poisson_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def test_poisson_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))
    """

def test_poisson_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))
    """
    
def test_poisson_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('PML')
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))
    """

def test_poisson_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('PML')
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=2, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))
    """
def test_poisson_sample_model():
    """
    Tests sampling function
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-1)
    """

def test_poisson_ppc():
    """
    Tests PPC value
    """
    """
    model = pf.GASLLT(data=data, family=pf.Poisson())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)
    """
