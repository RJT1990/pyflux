from .. import inference as ifr
from .. import output as op
import numpy as np
import scipy.stats as ss
from math import exp, sqrt, log, tanh
import math
from scipy import optimize
import matplotlib.pyplot as plt
from scores import *

class GAS(object):

	def __init__(self,dist,ar,sc,integ,data):
		self.dist = dist
		self.ar = ar
		self.sc = sc
		self.int = integ

		# Difference data
		for order in range(self.int):
			X = np.diff(data)
		self.data = X

		self.params = []

	def likelihood(self,beta,x,log):
		Y = np.array(x[max(self.ar,self.sc):len(x)])
		theta = np.ones(len(Y))*beta[0]
		var = np.ones(len(Y))
		scores = np.zeros(len(Y))

		# Loop over time series
		for t in range(max(self.ar,self.sc),len(Y)):

			# BUILD MEAN PREDICTION
			theta[t] = beta[0]

			# Loop over AR terms
			for ar_term in range(self.ar):
				#theta[t] += beta[2+ar_term]*theta[t-ar_term-1]
				theta[t] += theta[t-ar_term-1]

			# Loop over Score terms
			for sc_term in range(self.sc):
				theta[t] += tanh(beta[2+self.ar+sc_term])*scores[t-sc_term-1]

			# BUILD VARIANCE PREDICTION
			var[t] = exp(beta[1])

			scores[t] = lik_score(Y[t],theta[t],var[t],self.dist)

		if self.dist == "Laplace":
			if log == True:
				return -sum(ss.laplace.logpdf(theta,loc=Y,scale=var))
			else:
				return sum(ss.laplace.pdf(theta,loc=Y,scale=var))

	def fit(self,printer=True):
		# Difference dependent variable
		X = self.data

		phi = np.zeros(self.ar+self.sc+2)
		phi[1] = 1
		p = optimize.minimize(self.likelihood,phi,args=(X,True),method='L-BFGS-B')
		self.params = p.x
		p = optimize.minimize(self.likelihood,p.x,args=(X,True),method='BFGS')
		p_std = np.diag(p.hess_inv)**0.5

		print p