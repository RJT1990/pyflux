from .. import inference as ifr
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import distributions as dst
import numpy as np
import pandas as pd
import scipy.stats as ss
import copy
from scores import *
from math import exp, sqrt, log, tanh
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from .. import covariances as cov
import numdifftools as nd

class GAS(tsm.TSM):

	def __init__(self,data,dist,ar,sc,integ=0,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'GAS')

		# Parameters

		self.dist = dist
		self.ar = ar
		self.sc = sc
		self.integ = integ
		self.param_no = self.ar + self.sc + 1

		# Target variable transformation

		if self.dist in ['Normal','Laplace']:
			self.link = np.array
			self.scale = True
			self.param_no += 1
		elif self.dist in ['Poisson','Exponential']:
			self.link = np.exp
			self.scale = False

		# Check pandas or numpy

		if isinstance(data, pd.DataFrame):
			self.index = data.index			
			if target is None:
				self.data = data.ix[:,0].values
				self.data_name = data.columns.values[0]
			else:
				self.data = data[target]
				self.data_name = target					
			self.data_type = 'pandas'
			print str(self.data_name) + " picked as target variable"

		elif isinstance(data, np.ndarray):
			self.data_name = "Series"		
			self.data_type = 'numpy'	
			if any(isinstance(i, np.ndarray) for i in data):
				if target is None:
					self.data = data[0]
					self.index = range(len(data[0]))
				else:
					self.data = data[target]
					self.index = range(len(data[target]))
				print "Nested list " + str(target) + " chosen as target variable"
			else:
				self.data = data
				self.index = range(len(data))

		else:
			raise Exception("The data input is not pandas or numpy compatible!")

		# Difference data
		X = self.data
		for order in range(self.integ):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		

		self.param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None), 'q': dst.Normal(0,3)})		
		# AR priors
		for j in range(1,self.ar+1):
			self.param_desc.append({'name' : 'AR(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform='tanh'), 'q': dst.Normal(0,3)})
		# MA priors
		for k in range(self.ar+1,self.ar+self.sc+1):
			self.param_desc.append({'name' : 'SC(' + str(k-self.sc) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform='tanh'), 'q': dst.Normal(0,3)})
		
		# If the distribution has a scale parameter
		if self.scale is True:
			self.param_desc.append({'name' : 'Scale','index': self.ar+self.sc+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})

		# Other attributes

		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

	def model(self,beta,x):
		Y = np.array(x[max(self.ar,self.sc):len(x)])
		theta = np.ones(len(Y))*beta[0]
		scores = np.zeros(len(Y))
 
		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# Check if model has scale parameter
		if self.scale is True:
			model_scale = parm[len(parm)-1]
		else:
			model_scale = 0

		# Loop over time series
		for t in range(max(self.ar,self.sc),len(Y)):
			# BUILD MEAN PREDICTION
			theta[t] = parm[0]

			# Loop over AR terms
			for ar_term in range(self.ar):
				theta[t] += parm[1+ar_term]*theta[t-ar_term-1]

			# Loop over Score terms
			for sc_term in range(self.sc):
				theta[t] += parm[1+self.ar+sc_term]*scores[t-sc_term-1]

			scores[t] = lik_score(Y[t],self.link(theta[t]),model_scale,self.dist)

		return theta, Y, scores

	def likelihood(self,beta):
		theta, Y, scores = self.model(beta,self.data)

		if self.dist == "Laplace":
			return -np.sum(ss.laplace.logpdf(Y,loc=theta,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))
		elif self.dist == "Normal":
			return -np.sum(ss.norm.logpdf(Y,loc=theta,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))	
		elif self.dist == "Poisson":
			return -np.sum(ss.poisson.logpmf(Y,self.link(theta)))
		elif self.dist == "Exponential":
			return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))
	
	def predict(self,T=5,lookback=20,intervals=True):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Get data
			exp_values = self.data[max(self.ar,self.sc):len(self.data)]
			date_index = self.index[max(self.ar,self.sc):len(self.data)]
			
			for t in range(T):
				if self.data_type == 'pandas':
					date_index += pd.DateOffset(1)
				elif self.data_type == 'numpy':
					date_index.append(date_index[len(date_index)-1]+1)

			mu, Y, scores = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Expectation
			mu_exp = mu
			for t in range(T):
				new_value = params_to_use[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += params_to_use[j]*mu_exp[len(mu_exp)-j]

				if self.sc != 0:
					for k in range(1,self.sc+1):
						new_value += params_to_use[k+self.ar]*scores[len(mu_exp)-k]

				exp_values = np.append(exp_values,[self.link(new_value)])
				mu_exp = np.append(mu_exp,[new_value]) # For indexing consistency
				scores = np.append(scores,[0]) # expectation of score is zero

			# Simulate error bars (do analytically in future)
			# Expectation
			sim_vector = np.zeros([15000,T])

			for n in range(15000):
				mu_exp = mu
				values = self.data[max(self.ar,self.sc):len(self.data)]
				for t in range(T):
					new_value = params_to_use[0]
					if self.ar != 0:
						for j in range(1,self.ar+1):
							new_value += params_to_use[j]*mu_exp[len(mu_exp)-j]
					if self.sc != 0:
						for k in range(1,self.sc+1):
							new_value += params_to_use[k+self.ar]*scores[len(mu_exp)-k]						
					if self.dist == "Normal":
						rnd_value = np.random.normal(new_value, params_to_use[len(params_to_use)-1], 1)[0]
					elif self.dist == "Laplace":
						rnd_value = np.random.laplace(new_value, params_to_use[len(params_to_use)-1], 1)[0]
					elif self.dist == "Poisson":
						rnd_value = np.random.poisson(self.link(new_value), 1)[0]
					elif self.dist == "Exponential":
						rnd_value = np.random.exponential(1/self.link(new_value), 1)[0]

					values = np.append(values,[rnd_value])
					mu_exp = np.append(mu_exp,[new_value]) # For indexing consistency

				sim_vector[n] = values[(len(values)-T):(len(values))]

			test = np.transpose(sim_vector)
			error_bars = [np.percentile(i,95) for i in test] - exp_values[(len(values)-T):(len(values))]
			error_bars_90 = [np.percentile(i,90) for i in test] - exp_values[(len(values)-T):(len(values))]
			forecasted_values = exp_values[(len(values)-T):(len(values))]
			exp_values = exp_values[len(exp_values)-T-lookback:len(exp_values)]
			date_index = date_index[len(date_index)-T-lookback:len(date_index)]
			
			if intervals == True:
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)			
			plt.plot(date_index,exp_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()
