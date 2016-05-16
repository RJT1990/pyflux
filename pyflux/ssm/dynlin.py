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
from patsy import dmatrices, dmatrix, demo_data

from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .kalman import *

class DynLin(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** DYNAMIC LINEAR REGRESSION MODEL ****

	Parameters
	----------

	formula : string
		patsy string describing the regression

	data : pd.DataFrame
		Field to specify the data that will be used
	"""

	def __init__(self,formula,data):

		# Initialize TSM object
		super(DynLin,self).__init__('DynLin')

		# Parameters
		self.max_lag = 0
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"
		self.model_name = "Dynamic Linear Regression"

		# Format the data
		self.is_pandas = True # This is compulsory for this model type
		self.data_original = data
		self.formula = formula
		self.y, self.X = dmatrices(formula, data)
		self.param_no = self.X.shape[1] + 1
		self.y_name = self.y.design_info.describe()
		self.X_names = self.X.design_info.describe().split(" + ")
		self.y = np.array([self.y]).ravel()
		self.data = self.y
		self.X = np.array([self.X])[0]
		self.index = data.index

		# Add parameter information

		self._param_desc.append({'name' : 'Sigma^2 irregular','index': 0, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		for parm in range(self.param_no-1):
			self._param_desc.append({'name' : 'Sigma^2 ' + self.X_names[parm],'index': 1+parm, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		# Starting Parameters for Estimation
		self.starting_params = np.zeros(self.param_no)	

	def _forecast_model(self,beta,Z,h):
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

		T, _, R, Q, H = self._ss_matrices(beta)
		return dl_univariate_kalman_fcst(self.data,Z,H,T,Q,R,0.0,h)

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

		return dl_univariate_kalman(data,Z,H,T,Q,R,0.0)

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

		T = np.identity(self.param_no-1)
		H = np.identity(1)*self._param_desc[0]['prior'].transform(beta[0])		
		Z = self.X
		R = np.identity(self.param_no-1)
		
		Q = np.identity(self.param_no-1)
		for i in range(0,self.param_no-1):
			Q[i][i] = self._param_desc[i+1]['prior'].transform(beta[i+1])

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
		_, _, _, F, v = self._model(self.y,beta)
		loglik = 0.0
		for i in range(0,self.y.shape[0]):
			loglik += np.linalg.slogdet(F[:,:,i])[1] + np.dot(v[i],np.dot(np.linalg.pinv(F[:,:,i]),v[i]))
		return -(-((self.y.shape[0]/2)*log(2*pi))-0.5*loglik.T[0].sum())

	def plot_predict(self,h=5,past_values=20,intervals=True,oos_data=None,**kwargs):		
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		past_values : int (default : 20)
			How many past observations to show on the forecast graph?

		intervals : Boolean
			Would you like to show 95% prediction intervals for the forecast?

		oos_data : pd.DataFrame
			Data for the variables to be used out of sample (ys can be NaNs)

		Returns
		----------
		- Plot of the forecast
		"""		

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Sort/manipulate the out-of-sample data
			_, X_oos = dmatrices(self.formula, oos_data)
			X_oos = np.array([X_oos])[0]
			full_X = self.X.copy()
			full_X = np.append(full_X,X_oos,axis=0)
			Z = full_X

			# Retrieve data, dates and (transformed) parameters			
			a, P = self._forecast_model(self.params,Z,h)
			smoothed_series = np.zeros(self.y.shape[0]+h)
			series_variance = np.zeros(self.y.shape[0]+h)
			for t in range(self.y.shape[0]+h):
				smoothed_series[t] = np.dot(Z[t],a[:,t])
				series_variance[t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + self._param_desc[0]['prior'].transform(self.params[0])	

			date_index = self.shift_dates(h)
			plot_values = smoothed_series[smoothed_series.shape[0]-h-past_values:smoothed_series.shape[0]]
			forecasted_values = smoothed_series[smoothed_series.shape[0]-h:smoothed_series.shape[0]]
			lower = forecasted_values - 1.98*np.power(series_variance[series_variance.shape[0]-h:series_variance.shape[0]],0.5)
			upper = forecasted_values + 1.98*np.power(series_variance[series_variance.shape[0]-h:series_variance.shape[0]],0.5)
			lower = np.append(plot_values[plot_values.shape[0]-h-1],lower)
			upper = np.append(plot_values[plot_values.shape[0]-h-1],upper)

			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

			plt.figure(figsize=figsize)
			if intervals == True:
				plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], lower, upper, alpha=0.2)			

			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.y_name)
			plt.xlabel("Time")
			plt.ylabel(self.y_name)
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
			date_index = date_index[0:self.y.shape[0]+1]

			if series_type == 'Smoothed':
				mu, V = self.smoothed_state(self.data,self.params)
			elif series_type == 'Filtered':
				mu, _, _, _, _ = self.model(self.data,self.params)
			else:
				mu, V = self.smoothed_state(self.data,self.params)

			# Create smoothed/filtered aggregate series
			_, Z, _, _, _ = self._ss_matrices(self.params)
			smoothed_series = np.zeros(self.y.shape[0])

			for t in range(0,self.y.shape[0]):
				smoothed_series[t] = np.dot(Z[t],mu[:,t])

			plt.figure(figsize=figsize)	
			
			plt.subplot(self.param_no+1, 1, 1)
			plt.title(self.y_name + " Raw and " + series_type)	
			plt.plot(date_index,self.data,label='Data')
			plt.plot(date_index,smoothed_series,label=series_type,c='black')
			plt.legend(loc=2)

			for coef in range(0,self.param_no-1):
				plt.subplot(self.param_no+1, 1, 2+coef)
				plt.title("Beta " + self.X_names[coef])	
				plt.plot(date_index,mu[coef,0:mu.shape[1]-1],label='Data')
				plt.legend(loc=2)				
			
			plt.subplot(self.param_no+1, 1, self.param_no+1)
			plt.title("Measurement Error")
			plt.plot(date_index,self.data-smoothed_series,label='Irregular')
			plt.legend(loc=2)	

			plt.show()	

	def predict(self,h=5,oos_data=None):		
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		oos_data : pd.DataFrame
			Data for the variables to be used out of sample (ys can be NaNs)

		Returns
		----------
		- pd.DataFrame with predictions
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Sort/manipulate the out-of-sample data
			_, X_oos = dmatrices(self.formula, oos_data)
			X_oos = np.array([X_oos])[0]
			full_X = self.X.copy()
			full_X = np.append(full_X,X_oos,axis=0)
			Z = full_X

			# Retrieve data, dates and (transformed) parameters			
			a, P = self._forecast_model(self.params,Z,h)
			smoothed_series = np.zeros(h)
			for t in range(h):
				smoothed_series[t] = np.dot(Z[self.y.shape[0]+t],a[:,self.y.shape[0]+t])

			date_index = self.shift_dates(h)

			result = pd.DataFrame(smoothed_series)
			result.rename(columns={0:self.y_name}, inplace=True)
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
			data1 = self.data_original.iloc[0:self.data_original.shape[0]-h+t,:]
			data2 = self.data_original.iloc[self.data_original.shape[0]-h+t:self.data_original.shape[0],:]
			x = DynLin(formula=self.formula,data=data1)
			x.fit(printer=False)
			if t == 0:
				predictions = x.predict(1,oos_data=data2)
			else:
				predictions = pd.concat([predictions,x.predict(h=1,oos_data=data2)])
		
		predictions.rename(columns={0:self.y_name}, inplace=True)
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
		plt.title(self.y_name)
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
		q_dist = ss.multivariate_normal([0.0, 0.0], Q)
		rnd_q = q_dist.rvs(self.data.shape[0]+1)

		# Generate a_t+ and y_t+
		a_plus = np.zeros((T.shape[0],self.data.shape[0]+1)) 
		a_plus[0,0] = np.mean(self.data[0:5])
		y_plus = np.zeros(self.data.shape[0])

		for t in range(0,self.data.shape[0]+1):
			if t == 0:
				a_plus[:,t] = np.dot(T,a_plus[:,t]) + rnd_q[t,:]
				y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]
			else:
				if t != self.data.shape[0]:
					a_plus[:,t] = np.dot(T,a_plus[:,t-1]) + rnd_q[t,:]
					y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]

		alpha_hat, _ = self.smoothed_state(self.data,beta)
		alpha_hat_plus, _ = self.smoothed_state(y_plus,beta)
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
		alpha, V = dl_univariate_KFS(data,Z,H,T,Q,R,0.0)
		return alpha, V