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


def get_arma_process(ar, ma=0, samples=100000):
    """
    Generates a realization of an ARMA(ar, ma) process.
    """
    # a small noise so that the coefficients are well determined
    noise = np.random.normal(scale=0.01, size=samples)

    if ar > 0:
        # when sum of coefficients is smaller than 1 the process is stationary
        coefficients_ar = 0.99 * np.ones(shape=ar) / ar
    else:
        coefficients_ar = []

    if ma > 0:
        coefficients_ma = 0.5 * np.ones(shape=ma)
    else:
        coefficients_ma = []

    # start at 1
    x = np.ones(samples)

    for i in range(max(ar, ma), samples):
        x[i] = np.sum(coefficients_ar[d] * x[i - 1 - d] for d in range(0, ar)) + \
               np.sum(coefficients_ma[d] * noise[i - 1 - d] for d in range(0, ma)) + noise[i]

    return x[max(ar, ma):]


def _test_ARMA(ar, ma):
    """
    Tests that a ARMA(ar, ma) model fits an ARMA(ar, ma) process.
    """
    data = get_arma_process(ar=ar, ma=ma)

    model = ARIMA(data=data, ar=ar, ma=ma)
    coefficients = model.fit().z.get_z_values(transformed=True)

    assert (len(coefficients) == ar + ma + 2)  # 1 constant + 1 for the noise scale

    # Constant coefficient 0 within 0.1
    assert (np.abs(coefficients[0]) < 0.1)

    # AR coefficients within 10%
    if ar > 0:
        expected_ar = 0.99 / ar  # same as used in `get_arma_process`
        for ar_i in range(ar):
            assert (np.abs(coefficients[1 + ar_i] - expected_ar) / expected_ar < 0.1)

    # MA coefficients within 10%
    if ma > 0:
        expected_ma = 0.5  # same as used in `get_arma_process`
        for ma_i in range(ar, ma):
            assert (np.abs(coefficients[1 + ma_i] - expected_ma) / expected_ma < 0.1)

    # Normal scale coefficient within 10%
    expected_noise = 0.01  # same as used in `get_arma_process`
    assert (np.abs(coefficients[-1] - expected_noise) / expected_noise < 0.1)


def test_AR1():
    """
    Tests that an AR(1) model fits an AR(1) process.
    """
    _test_ARMA(1, 0)


def test_AR2():
    """
    Tests that an AR(2) model fits an AR(2) process.
    """
    _test_ARMA(2, 0)


def test_AR3():
    """
    Tests that an AR(3) model fits an AR(3) process.
    """
    _test_ARMA(3, 0)


def test_MA1():
    _test_ARMA(0, 1)


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
