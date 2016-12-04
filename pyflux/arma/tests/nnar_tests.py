import numpy as np
import pyflux as pf

noise = np.random.normal(0,1,300)
data = np.zeros(300)

for i in range(1,len(data)):
    data[i] = 0.9*data[i-1] + noise[i]

def test_couple_terms():
    """
    Tests an ARIMA model with 1 AR and 1 MA term and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
    """
    Tests an ARIMA model with 1 AR and 1 MA term, integrated once, and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, integ=1, family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100, mini_batch=50, map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_predict_length():
    """
    Tests that the prediction dataframe length is equal to the number of steps h
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
    """
    Tests that the prediction IS dataframe length is equal to the number of steps h
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nans():
    """
    Tests that the predictions are not nans
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
    """
    Tests that the in-sample predictions are not nans
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test_predict_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    predictions = model.predict(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test_predict_is_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    predictions = model.predict_is(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))

def test_sample_model():
    """
    Tests sampling function
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI', iterations=100, map_start=False)
    sample = model.sample(nsims=40)
    assert(sample.shape[0]==40)
    assert(sample.shape[1]==len(data)-2)

def test_ppc():
    """
    Tests PPC value
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI', iterations=100, map_start=False)
    p_value = model.ppc(nsims=40)
    assert(0.0 <= p_value <= 1.0)


def test_couple_terms():
    """
    Tests an ARIMA model with 1 AR and 1 MA term and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
    """
    Tests an ARIMA model with 1 AR and 1 MA term, integrated once, and that
    the latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, integ=1, family=pf.Normal(), units=2, layers=1)
    x = model.fit(iterations=50,map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100, map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    """
    Tests an ARIMA model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=1)
    x = model.fit('BBVI',iterations=100, mini_batch=32, map_start=False)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=2)
    x = model.fit('BBVI',iterations=100, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.NNAR(data=data, ar=2, family=pf.Normal(), units=2, layers=2)
    x = model.fit('BBVI',iterations=100, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_predict_length():
    """
    Tests that the prediction dataframe length is equal to the number of steps h
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    assert(model.predict(h=5).shape[0] == 5)

def test2_predict_is_length():
    """
    Tests that the prediction IS dataframe length is equal to the number of steps h
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    assert(model.predict_is(h=5).shape[0] == 5)

def test2_predict_nans():
    """
    Tests that the predictions are not nans
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test2_predict_is_nans():
    """
    Tests that the in-sample predictions are not nans
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test2_predict_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    predictions = model.predict(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test2_predict_is_nonconstant():
    """
    We should not really have predictions that are constant (should be some difference)...
    This captures bugs with the predict function not iterating forward
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit(iterations=50,map_start=False)
    predictions = model.predict_is(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))

def test2_sample_model():
    """
    Tests sampling function
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit('BBVI', iterations=100, map_start=False)
    sample = model.sample(nsims=40)
    assert(sample.shape[0]==40)
    assert(sample.shape[1]==len(data)-2)

def test2_ppc():
    """
    Tests PPC value
    """
    model = pf.NNAR(data=data, ar=2,  family=pf.Normal(), units=2, layers=2)
    x = model.fit('BBVI', iterations=100, map_start=False)
    p_value = model.ppc(nsims=40)
    assert(0.0 <= p_value <= 1.0)

