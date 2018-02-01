import numpy as np
from pyflux.arma import ARIMA

noise = np.random.normal(0,1,100)
data = np.zeros(100)

for i in range(1,len(data)):
    data[i] = 0.9*data[i-1] + noise[i]

def test_no_terms():
    """
    Tests an ARIMA model with no AR or MA terms, and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = ARIMA(data=data, ar=0, ma=0)
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 2)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)


def get_ar_process(ar, samples=10000):
    """
    Generates a realization of an AR(ar) process.
    """
    # a small noise so that the coefficients are well determined
    noise = np.random.normal(scale=0.001, size=samples)

    # sum of coefficients must be smaller than 1 for the process to be non-stationary
    coefficients = 0.99*np.ones(shape=ar)/ar

    # start is 1
    x = np.ones(samples)

    for i in range(ar, len(x)):
        x[i] = np.sum(coefficients[d] * x[i - d - 1] for d in range(0, ar)) + noise[i]

    return x


def _test_AR(ar):
    """
    Tests that a AR(ar) model fits an AR(ar) process.
    """
    data = get_ar_process(ar=ar)

    model = ARIMA(data=data, ar=ar, ma=0)
    x = model.fit()
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert (len(lvs) == ar + 2)

    coefficients = x.z.get_z_values(transformed=True)

    # Constant coefficient within 10%
    assert (np.abs(coefficients[0]) < 0.1)

    expected = 0.99/ar

    # AR coefficients within 10%
    for ar_i in range(ar):
        assert (np.abs(coefficients[1 + ar_i] - expected) / expected < 0.1)

    # Normal scale coefficient within 10%
    assert (np.abs(coefficients[-1] - 0.001) / 0.001 < 0.1)


def test_AR1():
    """
    Tests that an AR(1) model fits an AR(1) process.
    """
    _test_AR(1)


def test_AR2():
    """
    Tests that an AR(2) model fits an AR(2) process.
    """
    _test_AR(2)


def test_AR3():
    """
    Tests that an AR(3) model fits an AR(3) process.
    """
    _test_AR(3)


def test_couple_terms():
    """
    Tests an ARIMA model with 1 AR and 1 MA term and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1)
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
    """
    Tests an ARIMA model with 1 AR and 1 MA term, integrated once, and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = ARIMA(data=data, ar=1, ma=1, integ=1)
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
    """
    Tests that the prediction dataframe length is equal to the number of steps h
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
    """
    Tests that the prediction IS dataframe length is equal to the number of steps h
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
    """
    Tests that the predictions are not nans
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
    """
    Tests that the in-sample predictions are not nans
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test_predict_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    predictions = model.predict(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test_predict_is_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test_predict_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_is_intervals():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('BBVI', iterations=100, quiet_progress=True)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_is_intervals_bbvi():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('BBVI', iterations=100, quiet_progress=True)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('M-H', nsims=200, quiet_progress=True)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_predict_is_intervals_mh():
    """
    Tests prediction intervals are ordered correctly
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('M-H', nsims=200, quiet_progress=True)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values > predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values > predictions[model.data_name].values))
    assert(np.all(predictions[model.data_name].values > predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values > predictions['1% Prediction Interval'].values))

def test_sample_model():
    """
    Tests sampling function
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('BBVI', iterations=100, quiet_progress=True)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-2)

def test_ppc():
    """
    Tests PPC value
    """
    model = ARIMA(data=data, ar=2, ma=2)
    x = model.fit('BBVI', iterations=100, quiet_progress=True)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)
