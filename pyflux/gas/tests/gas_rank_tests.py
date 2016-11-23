import numpy as np
import pyflux as pf
import pandas as pd

data = pd.read_csv("http://www.pyflux.com/notebooks/nfl_data_new.csv")
data["PointsDiff"] = data["HomeScore"] - data["AwayScore"]

def test_mle():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_mh():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    x = model.fit('M-H', nsims=200)
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_pml():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_laplace():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_bbvi():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    x = model.fit('BBVI', iterations=100)
    assert(len(model.latent_variables.z_list) == 3)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_predict():
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    model.fit()
    prediction = model.predict("Denver Broncos","Carolina Panthers",neutral=True)
    assert(len(prediction[np.isnan(prediction)]) == 0)

def test_mle_two_components():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.Normal())
    model.add_second_component("HQB","AQB")
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    prediction = model.predict("Denver Broncos","Carolina Panthers","Peyton Manning","Cam Newton",neutral=True)
    assert(len(prediction[np.isnan(prediction)]) == 0)

def test_t_mle():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    x = model.fit()
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_t_pml():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    x = model.fit('PML')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_t_laplace():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    x = model.fit('Laplace')
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_t_bbvi():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    x = model.fit('BBVI', iterations=100)
    assert(len(model.latent_variables.z_list) == 4)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)

def test_t_predict():
    model = pf.GASRank(data=data,team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    model.fit()
    prediction = model.predict("Denver Broncos","Carolina Panthers",neutral=True)
    assert(len(prediction[np.isnan(prediction)]) == 0)

def test_t_mle_two_components():
    """
    Tests latent variable list length is correct, and that the estimated
    latent variables are not nan
    """
    model = pf.GASRank(data=data.iloc[0:300,:],team_1="HomeTeam", team_2="AwayTeam",
                   score_diff="PointsDiff", family=pf.t())
    model.add_second_component("HQB","AQB")
    x = model.fit('BBVI',iterations=50,map_start=False)
    assert(len(model.latent_variables.z_list) == 5)
    lvs = np.array([i.value for i in model.latent_variables.z_list])
    assert(len(lvs[np.isnan(lvs)]) == 0)
    prediction = model.predict("Denver Broncos","Carolina Panthers","Peyton Manning","Cam Newton",neutral=True)
    assert(len(prediction[np.isnan(prediction)]) == 0)

