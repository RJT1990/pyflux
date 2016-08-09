import numpy as np
import pandas as pd
import pyflux as pf

q = pf.distributions.q_Normal(loc=0,scale=0)

def test_q_sim_length():
	assert(len(q.sim(100))==100)

def test_q_loc_score():
	assert(q.loc_score(2)==2)

def test_q_scale_score():
	assert(q.scale_score(2)==3)

def test_q_score():
	assert(q.score(2,0)==2)
	assert(q.score(2,1)==3)

def test_q_params():
	assert(q.return_param(0)==0)
	assert(q.return_param(1)==0)

def test_logpdf():
	assert(q.logpdf(0)==np.log(1.0/np.sqrt(1.0*2.0*np.pi)))

def test_change_params():
	q.change_param(0,1)
	q.change_param(1,2)
	assert(q.loc==1)
	assert(q.scale==2)


def test_skewt_rvs():
	rvs = pf.distributions.skewt.rvs(4.0,0.9,100)
	assert(len(rvs)==100)

