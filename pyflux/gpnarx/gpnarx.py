from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import multivariate_normal
from math import exp, sqrt, log, tanh, pi
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from kernels import *

class GPNARX(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** GAUSSIAN PROCESS NONLINEAR AUTOREGRESSIVE (GP-NARX) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	ar : int
		Field to specify how many AR terms the model will have.

	kernel_type : str
		One of SE (SquaredExponential), OU (Ornstein-Uhlenbeck), RQ
		(RationalQuadratic). Defines kernel choice for GP-NAR.

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,ar,kernel_type='SE',integ=0,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'GPNARX')

		# Parameters
		self.ar = ar
		self.integ = integ

		if kernel_type == 'ARD':
			self.param_no = 3 + self.ar
		elif kernel_type == 'RQ':
			self.param_no = 3 + 1
		else:
			self.param_no = 3

		self.max_lag = self.ar
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"		
		self.kernel_type = kernel_type
		self.denom = 1.0 # Redundant - remove in future version; used for normalization

		# Check pandas or numpy
		if isinstance(data, pd.DataFrame):
			self.index = data.index			
			if target is None:
				self.data = data.ix[:,0].values
				self.data_name = data.columns.values[0]
			else:
				self.data = (data[target] - np.mean(data[target])) / np.power(np.std(data[target]),self.denom)				
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
		Y = self.data
		for order in range(self.integ):
			Y = np.diff(Y)
			self.data_name = "Differenced " + self.data_name
		self.index = self.index[self.integ:len(self.index)]

		# Apply normalization
		self.data_orig = Y		
		self.data = np.array(self.data_orig[self.max_lag:len(self.data_orig)]) # adjust for lags
		self.norm_mean = np.mean(self.data)
		self.norm_std = np.std(self.data)	
		self.data = (self.data - self.norm_mean) / np.power(self.norm_std,self.denom)

		# Define parameters

		self.param_desc.append({'name' : 'Noise Sigma','index': 0, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})

		self.param_desc.append({'name' : 'l','index': 1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})

		if self.kernel_type == 'SE':
			self.kernel = SquaredExponential(self.X(),1,1)
		elif self.kernel_type == 'OU':
			self.kernel = OrnsteinUhlenbeck(self.X(),1,1)
		elif self.kernel_type == 'RQ':
			self.param_desc.append({'name' : 'alpha','index': len(self.param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})
			self.kernel = RationalQuadratic(self.X(),1,1,1)

		self.param_desc.append({'name' : 'tau', 'index': len(self.param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})


	def start_params(self,beta):
		""" Transforms parameters for use in kernels

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		None (changes data in self.kernel)
		"""

		self.kernel.l = self.param_desc[1]['prior'].transform(beta[1])
		if self.kernel_type == 'RQ':
			self.kernel.a = self.param_desc[2]['prior'].transform(beta[2])			
			self.kernel.tau = self.param_desc[3]['prior'].transform(beta[3])	
		else:
			self.kernel.tau = self.param_desc[2]['prior'].transform(beta[2])			

	def X(self):
		""" Creates design matrix of variables to use in GP regression

		Returns
		----------
		The design matrix
		"""		

		# AR terms
		for i in xrange(0,self.ar):
			datapoint = self.data_orig[(self.max_lag-i-1):(len(self.data_orig)-i-1)]			
			if i==0:
				X = (datapoint - self.norm_mean)/np.power(self.norm_std,self.denom)
			else:
				X = np.vstack((X,(datapoint - self.norm_mean)/np.power(self.norm_std,self.denom)))
		return X

	def L(self,beta):
		""" Creates cholesky decomposition of covarianc ematrix

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		The cholesky decomposition (L) of K
		"""	

		return np.linalg.cholesky(self.kernel.K()) + np.identity(self.kernel.K().shape[0])*self.param_desc[0]['prior'].transform(beta[0])

	def alpha(self,beta):
		""" Covariance-derived term to construct expectations. See Rasmussen & Williams.

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		The alpha matrix/vector
		"""		

		L = self.L(beta)
		return np.linalg.solve(np.transpose(L),np.linalg.solve(L,np.transpose(self.data)))

	def E_fstar(self,beta):
		""" Expected values of the function given the covariance matrix and hyperparameters

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		The expected values of the function
		"""		

		self.start_params(beta)
		return np.dot(np.transpose(self.kernel.K()),self.alpha(beta))

	def v(self,beta):
		""" Covariance term used for variance of function (currently not in use)

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		Covariance term v.
		"""

		v = np.linalg.solve(self.L(beta),self.kernel.K())
		return v

	def var_fstar(self,beta):
		""" Covariance matrix for the estimated function

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		Covariance matrix for the estimated function 
		"""		

		self.start_params(beta)
		return self.kernel.K() - np.dot(np.dot(np.transpose(self.kernel.K()),np.linalg.pinv(self.kernel.K() + np.identity(self.kernel.K().shape[0])*self.param_desc[0]['prior'].transform(beta[0]))),self.kernel.K())		

	def pfit(self,beta,intervals=True,**kwargs):
		""" Plots the fit of the Gaussian process model to the data

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		intervals : Boolean
			Whether to plot uncertainty intervals or not

		Returns
		----------
		None (plots the fit of the function)
		"""

		figsize = kwargs.get('figsize',(10,7))

		date_index = self.index[self.max_lag:len(self.index)]
		expectation = self.E_fstar(self.params)
		posterior = multivariate_normal(expectation,self.var_fstar(self.params),allow_singular=True)
		simulations = 500
		sim_vector = np.zeros([simulations,len(expectation)])

		for i in range(simulations):
			sim_vector[i] = posterior.rvs()

		error_bars = []
		for pre in range(5,100,5):
			error_bars.append([(np.percentile(i,pre)*self.norm_std + self.norm_mean) for i in sim_vector.transpose()] - (expectation*self.norm_std + self.norm_mean))

		plt.figure(figsize=figsize)	

		plt.subplot(2, 2, 1)
		plt.title(self.data_name + " Raw")	


		plt.plot(date_index,self.data*self.norm_std + self.norm_mean,'k')

		plt.subplot(2, 2, 2)

		plt.title(self.data_name + " Raw and Expected")	
		plt.plot(date_index,self.data*self.norm_std + self.norm_mean,'k',alpha=0.2)
		plt.plot(date_index,self.E_fstar(beta)*self.norm_std + self.norm_mean,'b')

		plt.subplot(2, 2, 3)

		plt.title(self.data_name + " Raw and Expected (with intervals)")	

		if intervals == True:
			alpha =[0.15*i/float(100) for i in range(50,12,-2)]
			for count, pre in enumerate(error_bars):
				plt.fill_between(date_index, (expectation*self.norm_std + self.norm_mean)-pre, (expectation*self.norm_std + self.norm_mean)+pre,alpha=alpha[count])		

		plt.plot(date_index,self.data*self.norm_std + self.norm_mean,'k',alpha=0.2)
		plt.plot(date_index,self.E_fstar(beta)*self.norm_std + self.norm_mean,'b')

		plt.subplot(2, 2, 4)

		plt.title("Expected " + self.data_name + " (with intervals)")	

		if intervals == True:
			alpha =[0.15*i/float(100) for i in range(50,12,-2)]
			for count, pre in enumerate(error_bars):
				plt.fill_between(date_index, (expectation*self.norm_std + self.norm_mean)-pre, (expectation*self.norm_std + self.norm_mean)+pre,alpha=alpha[count])		

		plt.plot(date_index,self.E_fstar(beta)*self.norm_std + self.norm_mean,'b')

		plt.show()

	def likelihood(self,beta):
		""" Creates the negative log marginal likelihood of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		The log marginal logliklihood of the model
		"""				
		self.start_params(beta)
		return -(-0.5*(np.dot(np.transpose(self.data),self.alpha(beta))) - np.sum(np.diag(self.L(beta))) - (len(self.data)/2)*log(2*pi))

	######### Functions for prediction #########

	def construct_predict(self,beta,h):	
		""" Creates h-step ahead forecasts for the Gaussian process

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		h: int
			How many steps ahead to forecast

		Returns
		----------
		- predictions
		- variance of predictions
		"""				

		Xstart = copy.deepcopy(self.X())
		Xstart = [i for i in Xstart]

		predictions = np.zeros(h)
		variances = np.zeros(h)

		for step in range(h):

			Xstar = []

			for lag in range(self.max_lag):

				if lag == 0:
					if step == 0:
						Xstar.append([self.data[len(self.data)-1]])
						Xstart[0] = np.append(Xstart[0],self.data[len(self.data)-1])
					else:
						Xstar.append([predictions[step-1]])
						Xstart[0] = np.append(Xstart[0],predictions[step-1])
				else:
					Xstar.append([Xstart[lag-1][len(Xstart[lag-1])-2]])
					Xstart[lag] = np.append(Xstart[lag],Xstart[lag-1][len(Xstart[lag-1])-2])

			Kstar = self.kernel.Kstar(np.transpose(np.array(Xstar)))

			predictions[step] = np.dot(np.transpose(Kstar),self.alpha(beta))
			variances[step] = self.kernel.Kstarstar(np.transpose(np.array(Xstar))) - np.dot(np.dot(np.transpose(self.kernel.Kstar(np.transpose(np.array(Xstar)))),np.linalg.pinv(self.kernel.K() + np.identity(self.kernel.K().shape[0])*self.param_desc[0]['prior'].transform(beta[0]))),self.kernel.Kstar(np.transpose(np.array(Xstar))))		
		return predictions, variances, predictions - 1.98*np.power(variances,0.5), predictions + 1.98*np.power(variances,0.5)

	def predict(self,h=5,past_values=20,intervals=True,**kwargs):

		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		past_values : int (default : 20)
			How many past observations to show on the forecast graph?

		intervals : Boolean
			Would you like to show 95% prediction intervals for the forecast?

		Returns
		----------
		- Plot of the forecast
		- Error bars, forecasted_values, plot_values, plot_index
		"""		

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			predictions, variance, lower, upper = self.construct_predict(self.params,h)	
			full_predictions = np.append(self.data,predictions)
			full_lower = np.append(self.data,lower)
			full_upper = np.append(self.data,upper)
			date_index = self.shift_dates(h)
			lower = np.append(full_predictions[len(date_index)-h+1],lower)
			upper = np.append(full_predictions[len(date_index)-h+1],upper)

			# Plot values (how far to look back)
			plot_values = full_predictions[len(full_predictions)-h-past_values:len(full_predictions)]*self.norm_std + self.norm_mean
			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

			plt.figure(figsize=figsize)
			if intervals == True:
				plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], lower*self.norm_std + self.norm_mean, upper*self.norm_std + self.norm_mean,alpha=0.5)			
			
			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()

			self.predictions = {'error_bars' : np.array([lower,upper]), 'forecasted_values' : predictions, 'plot_values' : plot_values, 'plot_index': plot_index}
