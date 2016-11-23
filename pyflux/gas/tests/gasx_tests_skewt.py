import numpy as np
import pandas as pd
import pyflux as pf

# Set up some data to use for the tests

noise = np.random.normal(0,1,400)
y = np.zeros(400)
x1 = np.random.normal(0,1,400)
x2 = np.random.normal(0,1,400)

for i in range(1,len(y)):
    y[i] = 0.9*y[i-1] + noise[i] + 0.1*x1[i] - 0.3*x2[i]

data = pd.DataFrame([y,x1,x2]).T
data.columns = ['y', 'x1', 'x2']

countdata = np.random.poisson(3,300)
x1 = np.random.normal(0,1,300)
x2 = np.random.normal(0,1,300)
data2 = pd.DataFrame([countdata,x1,x2]).T
data2.columns = ['y', 'x1', 'x2']

y_oos = np.random.normal(0,1,30)
x1_oos = np.random.normal(0,1,30)
x2_oos = np.random.normal(0,1,30)
countdata_oos = np.random.poisson(3,30)

data_oos = pd.DataFrame([y_oos,x1_oos,x2_oos]).T
data_oos.columns = ['y', 'x1', 'x2']

data2_oos = pd.DataFrame([countdata_oos,x1_oos,x2_oos]).T
data2_oos.columns = ['y', 'x1', 'x2']

model_1 = pf.GASX(formula="y ~ x1", data=data, ar=0, sc=0, family=pf.Skewt())
x_1 = model_1.fit()

model_2 = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
x_2 = model_2.fit()

model_3 = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, integ=1, family=pf.Skewt())
x_3 = model_3.fit()

model_4 = pf.GASX(formula="y ~ x1", data=data, ar=2, sc=2, family=pf.Skewt())
x_4 = model_4.fit()

model_b_1 = pf.GASX(formula="y ~ x1 + x2", data=data, ar=0, sc=0, family=pf.Skewt())
x_1 = model_b_1.fit()

model_b_2 = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
x_2 = model_b_2.fit()

model_b_3 = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, integ=1, family=pf.Skewt())
x_3 = model_b_3.fit()

model_b_4 = pf.GASX(formula="y ~ x1 + x2", data=data, ar=2, sc=2, family=pf.Skewt())
x_4 = model_b_4.fit()

def test_no_terms():
    """
    Tests the length of the latent variable vector for an GASX model
    with no AR or SC terms, and tests that the values are not nan
    """
    assert(len(model_1.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model_1.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms():
    """
    Tests the length of the latent variable vector for an GASX model
    with 1 AR and 1 SC term, and tests that the values are not nan
    """
    assert(len(model_2.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model_2.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_couple_terms_integ():
    """
    Tests the length of the latent variable vector for an GASX model
    with 1 AR and 1 SC term and integrated once, and tests that the 
    values are not nan
    """
    assert(len(model_3.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model_3.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
    """
    Tests an GASX model estimated with BBVI, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=100)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_mini_batch():
    """
    Tests an GASX model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test_mh():
    """
    Tests an GASX model estimated with Metropolis-Hastings, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

# test_laplace - reactivate in future

def test_pml():
    """
    Tests an GASX model estimated with PML, and tests that the latent variable
    vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_sample_model():
    """
    Tests sampling function
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-1)

def test_ppc():
    """
    Tests PPC value
    """
    model = pf.GASX(formula="y ~ x1", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)




## Try more than one predictor

def test2_no_terms():
    """
    Tests the length of the latent variable vector for an GASX model
    with no AR or SC terms, and two predictors, and tests that the values 
    are not nan
    """
    assert(len(model_b_1.latent_variables.z_list) == 6)
    lvs = np.array([i.value for i in model_b_1.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_couple_terms():
    """
    Tests the length of the latent variable vector for an GASX model
    with 1 AR and 1 SC term, and two predictors, and tests that the values 
    are not nan
    """
    assert(len(model_b_2.latent_variables.z_list) == 8)
    lvs = np.array([i.value for i in model_b_2.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_bbvi():
    """
    Tests an GASX model estimated with BBVI, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500)
    assert(len(model.latent_variables.z_list) == 8)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_bbvi_mini_batch():
    """
    Tests an GASX model estimated with BBVI and that the length of the latent variable
    list is correct, and that the estimated latent variables are not nan
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, mini_batch=32)
    assert(len(model.latent_variables.z_list) == 8)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_bbvi_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test2_bbvi_mini_batch_elbo():
    """
    Tests that the ELBO increases
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI',iterations=500, mini_batch=32, record_elbo=True, map_start=False)
    assert(x.elbo_records[-1]>x.elbo_records[0])

def test2_mh():
    """
    Tests an GASX model estimated with MEtropolis-Hastings, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('M-H',nsims=300)
    assert(len(model.latent_variables.z_list) == 8)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_laplace():
    """
    Tests an GASX model estimated with Laplace, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_pml():
    """
    Tests an GASX model estimated with PML, with multiple predictors, and 
    tests that the latent variable vector length is correct, and that value are not nan
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 7)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test2_sample_model():
    """
    Tests sampling function
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI', iterations=100)
    sample = model.sample(nsims=100)
    assert(sample.shape[0]==100)
    assert(sample.shape[1]==len(data)-1)

def test2_ppc():
    """
    Tests PPC value
    """
    model = pf.GASX(formula="y ~ x1 + x2", data=data, ar=1, sc=1, family=pf.Skewt())
    x = model.fit('BBVI', iterations=100)
    p_value = model.ppc()
    assert(0.0 <= p_value <= 1.0)

