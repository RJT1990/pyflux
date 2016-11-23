import numpy as np
import pandas as pd
import pyflux as pf

# Set up some data to use for the tests

noise = np.random.normal(0,1,250)
y = np.zeros(250)
x1 = np.random.normal(0,1,250)
x2 = np.random.normal(0,1,250)

for i in range(1,len(y)):
    y[i] = 0.9*y[i-1] + noise[i] + 0.1*x1[i] - 0.3*x2[i]

data = pd.DataFrame([y,x1,x2]).T
data.columns = ['y', 'x1', 'x2']

y_oos = np.random.normal(0,1,30)
x1_oos = np.random.normal(0,1,30)
x2_oos = np.random.normal(0,1,30)
data_oos = pd.DataFrame([y_oos,x1_oos,x2_oos]).T
data_oos.columns = ['y', 'x1', 'x2']


def test_normal_no_terms():
    """
    Tests the length of the latent variable vector for an GASReg model
    with no AR or MA terms, and tests that the values are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_normal_bbvi():
    """
    Tests an GASReg model estimated with BBVI, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_normal_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    
def test_normal_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_normal_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_normal_mh():
    """
    Tests an GASReg model estimated with Metropolis-Hastings, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_normal_laplace():
    """
    Tests an GASReg model estimated with Laplace approximation, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_normal_pml():
    """
    Tests an GASReg model estimated with PML, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_normal_predict_length():
    """
    Tests that the length of the predict dataframe is equal to no of steps h
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test_normal_predict_is_length():
    """
    Tests that the length of the predict IS dataframe is equal to no of steps h
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_normal_predict_nans():
    """
    Tests that the predictions are not NaNs
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
        oos_data=data_oos).values)]) == 0)

def test_normal_predict_is_nans():
    """
    Tests that the predictions in-sample are not NaNs
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))
    """
def test_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))
    """

def test_sample_model():
    """
    Tests sampling function
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data))

def test_ppc():
    """
    Tests PPC value
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)


## Try more than one predictor

def test2_normal_no_terms():
    """
    Tests the length of the latent variable vector for an GASReg model
    with no AR or MA terms, and two predictors, and tests that the values 
    are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_normal_bbvi():
    """
    Tests an GASReg model estimated with BBVI, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_normal_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    
def test2_normal_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test2_normal_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI',iterations=100, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test2_normal_mh():
    """
    Tests an GASReg model estimated with MEtropolis-Hastings, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_normal_normal():
    """
    Tests an GASReg model estimated with Laplace, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_normal_pml():
    """
    Tests an GASReg model estimated with PML, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_normal_predict_length():
    """
    Tests that the length of the predict dataframe is equal to no of steps h
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(model.predict(h=5, oos_data=data_oos).shape[0] == 5)

def test2_normal_predict_is_length():
    """
    Tests that the length of the predict IS dataframe is equal to no of steps h
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test2_normal_predict_nans():
    """
    Tests that the predictions are not NaNs
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5, oos_data=data_oos).values[np.isnan(model.predict(h=5, 
        oos_data=data_oos).values)]) == 0)

def test2_normal_predict_is_nans():
    """
    Tests that the predictions in-sample are not NaNs
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test2_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test2_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1  + x2", data=data, family=pf.Cauchy())
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test2_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, oos_data=data_oos, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test2_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test2_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))
    """
def test2_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    """
    model = pf.GASReg(formula="y ~ x1", data=data, family=pf.Cauchy())
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))
    """

def test2_sample_model():
    """
    Tests sampling function
    """
    model = pf.GASReg(formula="y ~ x1 + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data))

def test2_ppc():
    """
    Tests PPC value
    """
    model = pf.GASReg(formula="y ~ x1  + x2", data=data, family=pf.Cauchy())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)
