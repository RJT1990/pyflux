from math import exp, sqrt, log, tanh, pi
import copy
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .kalman import *

class LLEV(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** LOCAL LEVEL MODEL ****

	Parameters
	----------
	data : pd.DataFrame or np.array
		Field to specify the time series data that will be used.

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.array)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,integ=0,target=None):

		# Initialize TSM object
		super(LLEV,self).__init__('LLEV')

		# Parameters
		self.integ = integ
		self.param_no = 2
		self.max_lag = 0
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"
		self.model_name = "LLEV"

		# Format the data
		self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
		self.data_original = self.data

		# Difference data
		for order in range(self.integ):
			self.data = np.diff(self.data)
			self.data_name = "Differenced " + self.data_name

		# Add parameter information

		self._param_desc.append({'name' : 'Sigma^2 irregular','index': 0, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})
		self._param_desc.append({'name' : 'Sigma^2 level','index': 1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		# Starting Parameters for Estimation
		self.starting_params = np.zeros(self.param_no)	

	def _forecast_model(self,beta,h):
		""" Creates forecasted states and variances

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		a : np.ndarray
			Forecasted states

		P : np.ndarray
			Variance of forecasted states
		"""		

		T, Z, R, Q, H = self._ss_matrices(beta)
		return univariate_kalman_fcst(self.data,Z,H,T,Q,R,0.0,h)

	def _model(self,data,beta):
		""" Creates the structure of the model

		Parameters
		----------
		data : np.array
			Contains the time series

		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		a,P,K,F,v : np.array
			Filted states, filtered variances, Kalman gains, F matrix, residuals
		"""		

		T, Z, R, Q, H = self._ss_matrices(beta)

		return univariate_kalman(data,Z,H,T,Q,R,0.0)

	def _ss_matrices(self,beta):
		""" Creates the state space matrices required

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		T, Z, R, Q, H : np.array
			State space matrices used in KFS algorithm
		"""		

		T = np.identity(1)
		R = np.identity(1)
		Z = np.identity(1)
		H = np.identity(1)*self._param_desc[0]['prior'].transform(beta[0])
		Q = np.identity(1)*self._param_desc[1]['prior'].transform(beta[1])

		return T, Z, R, Q, H

	def likelihood(self,beta):
		""" Creates the negative log marginal likelihood of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		The negative log logliklihood of the model
		"""			
		_, _, _, F, v = self._model(self.data,beta)
		loglik = 0.0
		for i in range(0,self.data.shape[0]):
			loglik += np.linalg.slogdet(F[:,:,i])[1] + np.dot(v[i],np.dot(np.linalg.pinv(F[:,:,i]),v[i]))
		return -(-((self.data.shape[0]/2)*log(2*pi))-0.5*loglik.T[0].sum())

	def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):		
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
		"""		

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Retrieve data, dates and (transformed) parameters			
			a, P = self._forecast_model(self.params,h)
			date_index = self.shift_dates(h)
			plot_values = a[0][a[0].shape[0]-h-past_values:a[0].shape[0]]
			forecasted_values = a[0][a[0].shape[0]-h:a[0].shape[0]]
			lower = forecasted_values - 1.98*np.power(P[0][0][P[0][0].shape[0]-h:P[0][0].shape[0]] + self._param_desc[0]['prior'].transform(self.params[0]),0.5)
			upper = forecasted_values + 1.98*np.power(P[0][0][P[0][0].shape[0]-h:P[0][0].shape[0]] + self._param_desc[0]['prior'].transform(self.params[0]),0.5)
			lower = np.append(plot_values[plot_values.shape[0]-h-1],lower)
			upper = np.append(plot_values[plot_values.shape[0]-h-1],upper)

			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

			plt.figure(figsize=figsize)
			if intervals == True:
				plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], lower, upper, alpha=0.2)			

			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()

	def plot_fit(self,**kwargs):
		""" Plots the fit of the model

		Returns
		----------
		None (plots data and the fit)
		"""

		figsize = kwargs.get('figsize',(10,7))
		series_type = kwargs.get('series_type','Smoothed')

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			date_index = copy.deepcopy(self.index)
			date_index = date_index[self.integ:self.data_original.shape[0]+1]

			if series_type == 'Smoothed':
				mu = self.smoothed_state(self.data,self.params)
			elif series_type == 'Filtered':
				mu, _, _, _, _ = self._model(self.data,self.params)
			else:
				mu = self.smoothed_state(self.data,self.params)

			mu = mu[0][0:mu[0].shape[0]-1]
			plt.figure(figsize=figsize)	
			
			plt.subplot(3, 1, 1)
			plt.title(self.data_name + " Raw and " + series_type)	

			plt.plot(date_index,self.data,label='Data')
			plt.plot(date_index,mu,label=series_type,c='black')
			plt.legend(loc=2)
			
			plt.subplot(3, 1, 2)
			plt.title(self.data_name + " Local Level")	
			plt.plot(date_index,mu)
			
			plt.subplot(3, 1, 3)
			plt.title("Measurement Noise")	
			plt.plot(date_index,self.data-mu)
			plt.show()

	def predict(self,h=5):		
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		Returns
		----------
		- pd.DataFrame with predictions
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Retrieve data, dates and (transformed) parameters			
			a, P = self._forecast_model(self.params,h)
			date_index = self.shift_dates(h)
			forecasted_values = a[0][a[0].shape[0]-h:a[0].shape[0]]

			result = pd.DataFrame(forecasted_values)
			result.rename(columns={0:self.data_name}, inplace=True)
			result.index = date_index[len(date_index)-h:len(date_index)]

			return result

	def predict_is(self,h=5):
		""" Makes dynamic in-sample predictions with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps would you like to forecast?

		Returns
		----------
		- pd.DataFrame with predicted values
		"""		

		predictions = []

		for t in range(0,h):
			x = LLEV(integ=self.integ,data=self.data_original[0:(self.data_original.shape[0]-h+t)])
			x.fit(printer=False)
			if t == 0:
				predictions = x.predict(1)
			else:
				predictions = pd.concat([predictions,x.predict(1)])
		
		predictions.rename(columns={0:self.data_name}, inplace=True)
		predictions.index = self.index[(len(self.index)-h):len(self.index)]

		return predictions

	def plot_predict_is(self,h=5,**kwargs):
		""" Plots forecasts with the estimated model against data
			(Simulated prediction with data)

		Parameters
		----------
		h : int (default : 5)
			How many steps to forecast

		Returns
		----------
		- Plot of the forecast against data 
		"""		

		figsize = kwargs.get('figsize',(10,7))

		plt.figure(figsize=figsize)
		predictions = self.predict_is(h)
		data = self.data[(len(self.data)-h):len(self.data)]
		plt.plot(predictions.index,data,label='Data')
		plt.plot(predictions.index,predictions,label='Predictions',c='black')
		plt.title(self.data_name)
		plt.legend(loc=2)	
		plt.show()			

	def simulation_smoother(self,beta):
		""" Koopman's simulation smoother - simulates from states given
		model parameters and observations

		Parameters
		----------

		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		- A simulated state evolution
		"""			

		T, Z, R, Q, H = self._ss_matrices(beta)

		# Generate e_t+ and n_t+
		rnd_h = np.random.normal(0,np.sqrt(H),self.data.shape[0]+1)
		q_dist = ss.multivariate_normal([0.0], Q)
		rnd_q = q_dist.rvs(self.data.shape[0]+1)

		# Generate a_t+ and y_t+
		a_plus = np.zeros((T.shape[0],self.data.shape[0]+1)) 
		a_plus[0,0] = np.mean(self.data[0:5])
		y_plus = np.zeros(self.data.shape[0])

		for t in range(0,self.data.shape[0]+1):
			if t == 0:
				a_plus[:,t] = np.dot(T,a_plus[:,t]) + rnd_q[t]
				y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]
			else:
				if t != self.data.shape[0]:
					a_plus[:,t] = np.dot(T,a_plus[:,t-1]) + rnd_q[t]
					y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]

		alpha_hat = self.smoothed_state(self.data,beta)
		alpha_hat_plus = self.smoothed_state(y_plus,beta)
		alpha_tilde = alpha_hat - alpha_hat_plus + a_plus
	
		return alpha_tilde

	def smoothed_state(self,data,beta):
		""" Creates the negative log marginal likelihood of the model

		Parameters
		----------

		data : np.array
			Data to be smoothed

		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		- Smoothed states
		"""			

		T, Z, R, Q, H = self._ss_matrices(beta)
		alpha, V = univariate_KFS(data,Z,H,T,Q,R,0.0)
		return alpha