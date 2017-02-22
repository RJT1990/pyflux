import numpy as np
import pandas as pd
from pyflux.arma import ARIMAX
from pyflux.families import Exponential

# Set up some data to use for the tests

countdata = np.random.exponential(3,150)
x1 = np.random.normal(0,1,150) + countdata*0.10
x2 = np.random.normal(0,1,150) - countdata*0.25
data = pd.DataFrame([countdata,x1,x2]).T
data.columns = ['y', 'x1', 'x2']

x1_oos = np.random.normal(0,1,30)
x2_oos = np.random.normal(0,1,30)
countdata_oos = np.random.exponential(3,30)

data_oos = pd.DataFrame([countdata_oos,x1_oos,x2_oos]).T
data_oos.columns = ['y', 'x1', 'x2']


def a_test_no_terms():
    """
    Tests the length of the latent variable vector for an ARIMAX model
    with no AR or MA terms, and tests that the values are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=0, ma=0, family=Exponential())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_couple_terms():
    """
    Tests the length of the latent variable vector for an ARIMAX model
    with 1 AR and 1 MA term, and tests that the values are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_couple_terms_integ():
    """
    Tests the length of the latent variable vector for an ARIMAX model
    with 1 AR and 1 MA term and integrated once, and tests that the 
    values are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, integ=1, family=Exponential())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_bbvi():
    """
    Tests an ARIMAX model estimated with BBVI, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=300, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def a_test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=300, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def a_test_mh():
    """
    Tests an ARIMAX model estimated with Metropolis-Hastings, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_laplace():
    """
    Tests an ARIMAX model estimated with Laplace approximation, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_pml():
    """
    Tests an ARIMAX model estimated with PML, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test_predict_length():
    """
    Tests that the length of the predict dataframe is equal to no of steps h
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def a_test_predict_is_length():
    """
    Tests that the length of the predict IS dataframe is equal to no of steps h
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def a_test_predict_nans():
    """
    Tests that the predictions are not NaNs
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
        oos_data=data_oos).values)]) == 0)

def a_test_predict_is_nans():
    """
    Tests that the predictions in-sample are not NaNs
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def a_test_predict_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=200)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def a_test_predict_is_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=200)
    predictions = model.predict_is(h=10, fit_method='BBVI', intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def a_test_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('M-H', nsims=1000)
    x.summary()
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test_sample_model():
    """
    Tests sampling function
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-2)

def a_test_ppc():
    """
    Tests PPC value
    """
    model = ARIMAX(formula="y ~ x1", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)











## Try more than one predictor

def a_test2_no_terms():
    """
    Tests the length of the latent variable vector for an ARIMAX model
    with no AR or MA terms, and two predictors, and tests that the values 
    are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=0, ma=0, family=Exponential())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_couple_terms():
    """
    Tests the length of the latent variable vector for an ARIMAX model
    with 1 AR and 1 MA term, and two predictors, and tests that the values 
    are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_bbvi():
    """
    Tests an ARIMAX model estimated with BBVI, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_bbvi():
    """
    Tests an ARIMAX model estimated with BBVI, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_mh():
    """
    Tests an ARIMAX model estimated with MEtropolis-Hastings, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_laplace():
    """
    Tests an ARIMAX model estimated with Laplace, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_pml():
    """
    Tests an ARIMAX model estimated with PML, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def a_test2_predict_length():
    """
    Tests that the length of the predict dataframe is equal to no of steps h
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def a_test2_predict_is_length():
    """
    Tests that the length of the predict IS dataframe is equal to no of steps h
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def a_test2_predict_nans():
    """
    Tests that the predictions are not NaNs
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
        oos_data=data_oos).values)]) == 0)

def a_test2_predict_is_nans():
    """
    Tests that the predictions in-sample are not NaNs
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)


def a_test2_predict_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=200)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def a_test2_predict_is_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=1, ma=1, family=Exponential())
    x = model.fit('BBVI',iterations=200)
    predictions = model.predict_is(h=10, fit_method='BBVI', intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def a_test2_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values >= predictions['1% Prediction Interval'].values))

def a_test2_sample_model():
    """
    Tests sampling function
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-2)

def a_test2_ppc():
    """
    Tests PPC value
    """
    model = ARIMAX(formula="y ~ x1 + x2", data=data, ar=2, ma=2, family=Exponential())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)


