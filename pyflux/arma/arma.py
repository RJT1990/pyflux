from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import exp, sqrt, log, tanh
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from .. import covariances as cov
import numdifftools as nd

class ARIMA(tsm.TSM):

	def __init__(self,data,ar,ma,integ=0,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'ARIMA')

		# Parameters
		self.ar = ar
		self.ma = ma
		self.integ = integ
		self.param_no = self.ar + self.ma + 2

		# Check Data format
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
			print ""

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
				print ""
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
			self.param_desc.append({'name' : 'AR(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# MA priors
		for k in range(self.ar+1,self.ar+self.ma+1):
			self.param_desc.append({'name' : 'MA(' + str(k-self.ar) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# Variance prior
		self.param_desc.append({'name' : 'Sigma','index': self.ar+self.ma+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})

		# Other attributes

		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

	# Holds the core model matrices
	def model(self,beta,x):
		Y = np.array(x[max(self.ar,self.ma):len(x)])
		X = np.ones(len(Y))

		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# AR terms
		for i in range(self.ar):
			X = np.vstack((X,x[(max(self.ar,self.ma)-i-1):(len(x)-i-1)]))

		mu = np.matmul(np.transpose(X),parm[0:len(parm)-1-self.ma])

		# MA terms
		if self.ma != 0:
			for t in range(max(self.ar,self.ma),len(Y)):
				for k in range(self.ma):
						mu[t] += parm[1+self.ar+k]*(Y[t-1-k]-mu[t-1-k])

		return mu, Y 

	# Returns negative log likelihood
	def likelihood(self,beta):
		mu, Y = self.model(beta,self.data)
		return -np.sum(ss.norm.logpdf(Y,loc=mu,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))
		
	# Produces T-step ahead forecast for the series
	# This code is very inefficient; needs amending
	def predict(self,T=5,lookback=20,intervals=True):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Get data
			exp_values = self.data[max(self.ar,self.ma):len(self.data)]
			date_index = self.index[max(self.ar,self.ma):len(self.data)]
			
			# Create future dates
			for t in range(T):
				if self.data_type == 'pandas':
					date_index += pd.DateOffset(1)
				elif self.data_type == 'numpy':
					date_index.append(date_index[len(date_index)-1]+1)

			# Grab implied model information using estimated parameters
			mu, Y = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Expected values of the series
			mu_exp = mu
			for t in range(T):
				new_value = params_to_use[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += params_to_use[j]*exp_values[len(exp_values)-j]

				if self.ma != 0:
					for k in range(1,self.ma+1):
						if (k-1) >= t:
							new_value += params_to_use[k+self.ar]*(exp_values[len(exp_values)-k]-mu_exp[len(exp_values)-k])

				exp_values = np.append(exp_values,[new_value])
				mu_exp = np.append(mu_exp,[0]) # For indexing consistency

			# Simulate error bars (do analytically in future)
			sim_vector = np.zeros([15000,T])

			for n in range(15000):
				mu_exp = mu
				values = self.data[max(self.ar,self.ma):len(self.data)]
				for t in range(T):
					new_value = params_to_use[0] + np.random.randn(1)*params_to_use[len(params_to_use)-1]

					if self.ar != 0:
						for j in range(1,self.ar+1):
							new_value += params_to_use[j]*values[len(values)-j]

					if self.ma != 0:
						for k in range(1,self.ma+1):
							if (k-1) >= t:
								new_value += params_to_use[k+self.ar]*(values[len(values)-k]-mu_exp[len(values)-k])

					values = np.append(values,[new_value])
					mu_exp = np.append(mu_exp,[0]) # For indexing consistency

				sim_vector[n] = values[(len(values)-T):(len(values))]

			test = np.transpose(sim_vector)
			error_bars = [np.percentile(i,95) for i in test] - exp_values[(len(values)-T):(len(values))]
			error_bars_90 = [np.percentile(i,90) for i in test] - exp_values[(len(values)-T):(len(values))]
			forecasted_values = exp_values[(len(values)-T):(len(values))]
			exp_values = exp_values[len(exp_values)-T-lookback:len(exp_values)]
			date_index = date_index[len(date_index)-T-lookback:len(date_index)]
			
			# Plot prediction graph
			if intervals is True:
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)			
			plt.plot(date_index,exp_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()





