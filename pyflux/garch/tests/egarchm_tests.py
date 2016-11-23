import numpy as np
import pyflux as pf
import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime

jpm = DataReader('JPM',  'yahoo', datetime(2014,1,1), datetime(2016,3,10))
data = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
data.index = jpm.index.values[1:jpm.index.values.shape[0]]
data.columns = ['JPM Returns']

def test_no_terms():
    model = pf.EGARCHM(data=data, p=0, q=0)
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('BBVI', map_start=False, iterations=100)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('BBVI',iterations=100, map_start=False, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('BBVI',iterations=300, map_start=False, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('BBVI',iterations=300, map_start=False, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_mh():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('M-H', nsims=300)
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
    model = pf.EGARCHM(data=data, p=1, q=1)
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict_length():
    model = pf.EGARCHM(data=data, p=2, q=2)
    x = model.fit()
    x.summary()
    assert(model.predict(h=5).shape[0] == 5)

def test_predict_is_length():
    model = pf.EGARCHM(data=data, p=2, q=2)
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_predict_nonconstant():
    model = pf.EGARCHM(data=data, p=2, q=2)
    x = model.fit()
    predictions = model.predict(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test_predict_is_nonconstant():
    model = pf.EGARCHM(data=data, p=2, q=2)
    x = model.fit()
    predictions = model.predict_is(h=5, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))

def test_predict_nans():
    model = pf.EGARCHM(data=data, p=2, q=2)
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_predict_is_nans():
    model = pf.EGARCHM(data=data, q=2, p=2)
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test_predict_intervals():
    model = pf.EGARCHM(data=data, q=2, p=2)
    x = model.fit()
    predictions = model.predict(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_predict_is_intervals():
    model = pf.EGARCHM(data=data, q=2, p=2)
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_predict_intervals_bbvi():
    model = pf.EGARCHM(data=data, q=1, p=1)
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_predict_is_intervals_bbvi():
    model = pf.EGARCHM(data=data, q=1, p=1)
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_predict_intervals_mh():
    model = pf.EGARCHM(data=data, q=1, p=1)
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_predict_is_intervals_mh():
    model = pf.EGARCHM(data=data, q=1, p=1)
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_sample_model():
    model = pf.EGARCHM(data=data, q=2, p=2)
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-2)

def test_ppc():
    model = pf.EGARCHM(data=data, q=2, p=2)
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)


def test_lev_no_terms():
    model = pf.EGARCHM(data=data, p=0, q=0)
    model.add_leverage()
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_couple_terms():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_bbvi():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('BBVI', iterations=100)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_bbvi_mini_batch():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('BBVI',iterations=100, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_bbvi_elbo():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('BBVI',iterations=300, map_start=False, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_lev_bbvi_mini_batch_elbo():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('BBVI',iterations=300, map_start=False, mini_batch=32, record_elbo=True)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_lev_mh():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('M-H', nsims=300)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_laplace():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_pml():
    model = pf.EGARCHM(data=data, p=1, q=1)
    model.add_leverage()
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_lev_predict_length():
    model = pf.EGARCHM(data=data, p=2, q=2)
    model.add_leverage()
    x = model.fit()
    x.summary()
    assert(model.predict(h=5).shape[0] == 5)

def test_lev_predict_is_length():
    model = pf.EGARCHM(data=data, p=2, q=2)
    model.add_leverage()
    x = model.fit()
    assert(model.predict_is(h=5).shape[0] == 5)

def test_lev_predict_nonconstant():
    model = pf.EGARCHM(data=data, p=2, q=2)
    model.add_leverage()
    x = model.fit()
    predictions = model.predict(h=10, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))
    
def test_lev_predict_is_nonconstant():
    model = pf.EGARCHM(data=data, p=2, q=2)
    model.add_leverage()
    x = model.fit()
    predictions = model.predict_is(h=5, intervals=False)
    assert(not np.all(predictions.values==predictions.values[0]))

def test_lev_predict_nans():
    model = pf.EGARCHM(data=data, p=2, q=2)
    model.add_leverage()
    x = model.fit()
    x.summary()
    assert(len(model.predict(h=5).values[np.isnan(model.predict(h=5).values)]) == 0)

def test_lev_predict_is_nans():
    model = pf.EGARCHM(data=data, q=2, p=2)
    model.add_leverage()
    x = model.fit()
    x.summary()
    assert(len(model.predict_is(h=5).values[np.isnan(model.predict_is(h=5).values)]) == 0)

def test_lev_predict_intervals():
    model = pf.EGARCHM(data=data, q=2, p=2)
    model.add_leverage()
    x = model.fit()
    predictions = model.predict(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_predict_is_intervals():
    model = pf.EGARCHM(data=data, q=2, p=2)
    model.add_leverage()
    x = model.fit()
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_predict_intervals_bbvi():
    model = pf.EGARCHM(data=data, q=1, p=1)
    model.add_leverage()
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict(h=10, intervals=True)

    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_predict_is_intervals_bbvi():
    model = pf.EGARCHM(data=data, q=1, p=1)
    model.add_leverage()
    x = model.fit('BBVI', iterations=100)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_predict_intervals_mh():
    model = pf.EGARCHM(data=data, q=1, p=1)
    model.add_leverage()
    x = model.fit('M-H', nsims=400)
    predictions = model.predict(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_predict_is_intervals_mh():
    model = pf.EGARCHM(data=data, q=1, p=1)
    model.add_leverage()
    x = model.fit('M-H', nsims=400)
    predictions = model.predict_is(h=10, intervals=True)
    assert(np.all(predictions['99% Prediction Interval'].values + 0.000001 >= predictions['95% Prediction Interval'].values))
    assert(np.all(predictions['95% Prediction Interval'].values + 0.000001 >= predictions['5% Prediction Interval'].values))
    assert(np.all(predictions['5% Prediction Interval'].values + 0.000001 >= predictions['1% Prediction Interval'].values))

def test_lev_sample_model():
    model = pf.EGARCHM(data=data, q=2, p=2)
    model.add_leverage()
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-2)

def test_lev_ppc():
    model = pf.EGARCHM(data=data, q=2, p=2)
    model.add_leverage()
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)
