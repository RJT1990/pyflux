from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import exp, sqrt, log, tanh, pi
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from kalman import *

class LLEV(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** LOCAL LEVEL MODEL ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,integ=0,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'LLEV')

		# Parameters
		self.integ = integ
		self.param_no = 2
		self.max_lag = 0
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"
		self.model_name = "LLEV"

		# Format the data
		self.data, self.data_name, self.data_type, self.index = dc.data_check(data,target)

		# Difference data
		X = self.data
		for order in range(self.integ):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		
		self.data_length = X
		self.cutoff = 0

		# Add parameter information

		self.param_desc.append({'name' : 'Sigma_eps','index': 0, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})
		self.param_desc.append({'name' : 'Sigma_eta','index': 1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})

	def model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		v : np.ndarray
			Contains the errors
		"""		
		T = np.identity(1)
		R = np.identity(1)
		Z = np.identity(1)
		H = np.identity(1)*self.param_desc[0]['prior'].transform(beta[0])
		Q = np.identity(1)*self.param_desc[1]['prior'].transform(beta[1])
		return univariate_kalman(self.data,Z,H,T,Q,R)

	def likelihood(self,beta):
		_, F, v = self.model(beta)

		loglik = 0
		for i in xrange(0,self.data.shape[0]):
			loglik += np.linalg.slogdet(F[:,:,i])[1] + np.dot(v[i],np.dot(np.linalg.pinv(F[:,:,i]),v[i]))
		return -(-((self.data.shape[0]/2)*log(2*pi))-0.5*loglik.T[0].sum())

	def forecast_model(self,beta,h):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		v : np.ndarray
			Contains the errors
		"""		
		T = np.identity(1)
		R = np.identity(1)
		Z = np.identity(1)
		H = np.identity(1)*self.param_desc[0]['prior'].transform(beta[0])
		Q = np.identity(1)*self.param_desc[1]['prior'].transform(beta[1])
		return univariate_kalman_fcst(self.data,Z,H,T,Q,R,h)


	def predict(self,h=5,past_values=20,intervals=True,**kwargs):

		figsize = kwargs.get('figsize',(10,7))
		
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		past_values : int (default : 20)
			How many past observations to show on the forecast graph?

		intervals : Boolean
			Would you like to show prediction intervals for the forecast?

		Returns
		----------
		- Plot of the forecast
		- Error bars, forecasted_values, plot_values, plot_index
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Retrieve data, dates and (transformed) parameters			
			a, P = self.forecast_model(self.params,h)
			date_index = self.shift_dates(h)

			plot_values = a[0][len(a[0])-h-past_values:len(a[0])]
			forecasted_values = a[0][len(a[0])-h:len(a[0])]
			lower = forecasted_values - 1.98*np.power(P[0][0][len(P[0][0])-h:len(P[0][0])],0.5)
			upper = forecasted_values + 1.98*np.power(P[0][0][len(P[0][0])-h:len(P[0][0])],0.5)
			lower = np.append(plot_values[len(plot_values)-h-1],lower)
			upper = np.append(plot_values[len(plot_values)-h-1],upper)

			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

			plt.figure(figsize=figsize)
			if intervals == True:
				plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], lower, upper, alpha=0.2)			

			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()

			self.predictions = {'error_bars' : np.array([lower,upper]), 'forecasted_values' : forecasted_values, 'plot_values' : plot_values, 'plot_index': plot_index}


	def plot_fit(self,**kwargs):
		""" Plots the fit of the model

		Returns
		----------
		None (plots data and the fit)
		"""

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			date_index = self.index
			mu, _, _ = self.model(self.params)
			mu = mu[0][0:len(mu[0])-1]
			plt.figure(figsize=figsize)	
			
			plt.subplot(3, 1, 1)
			plt.title(self.data_name + " Raw and Filtered")	

			plt.plot(date_index,self.data,label='Data')
			plt.plot(date_index,mu,label='Filter',c='black')
			plt.legend(loc=2)
			
			plt.subplot(3, 1, 2)
			plt.title(self.data_name + " Local Level")	
			plt.plot(date_index,mu)
			
			plt.subplot(3, 1, 3)
			plt.title("Measurement Noise")	
			plt.plot(date_index,self.data-mu)
			plt.show()