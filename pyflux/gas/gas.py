from math import exp, sqrt, log, tanh
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
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import distributions as dst
from .. import data_check as dc

from .scores import *

class GAS(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.array
		Field to specify the time series data that will be used.

	dist : str
		Field to specify the distribution. Options include 'Normal',
		'Laplace', 'Poisson', 'Exponential' and 't'.

	ar : int
		Field to specify how many AR terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		fitting methods (MLE/MAP).

	sc : int
		Field to specify how many Score terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		fitting methods (MLE/MAP).

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.array)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,dist,ar,sc,integ=0,target=None):

		# Initialize TSM object		
		super(GAS,self).__init__('GAS')

		self.dist = dist
		self.ar = ar
		self.sc = sc
		self.integ = integ
		self.model_name = "GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ")"
		self.param_no = self.ar + self.sc + 1
		self.max_lag = max(self.ar,self.sc)
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

		# Target variable transformation
		if self.dist in ['Normal','Laplace']:
			self.link = np.array
			self.scale = True
			self.param_no += 1
		elif self.dist in ['Poisson','Exponential']:
			self.link = np.exp
			self.scale = False
		elif self.dist == 't':
			self.link = np.array
			self.scale = True
			self.param_no +=2

		# Format the data
		self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
		self.data_original = self.data.copy()

		# Difference data
		for order in range(0,self.integ):
			self.data = np.diff(self.data)
			self.data_name = "Differenced " + self.data_name

		# Add parameter information

		self._param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None), 'q': dst.q_Normal(0,3)})		
		
		# AR priors
		
		for j in range(1,self.ar+1):
			self._param_desc.append({'name' : 'AR(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.q_Normal(0,3)})
		
		for k in range(self.ar+1,self.ar+self.sc+1):
			self._param_desc.append({'name' : 'SC(' + str(k-self.ar) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.q_Normal(0,3)})
	
		if self.dist == 't':
			self._param_desc.append({'name' : 'v','index': len(self._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})			

		if self.scale is True:
			self._param_desc.append({'name' : 'Scale','index': self.ar+self.sc+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		# Starting Parameters for Estimation
		self.starting_params = np.zeros(self.param_no)
		if self.dist in ['Laplace','Normal']:
			self.starting_params[0] = np.mean(self.data)
		elif self.dist == 'Poisson':
			self.starting_params[0] = np.log(np.mean(self.data))
			
	def _model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		theta : np.array
			Contains the predicted values for the time series

		Y : np.array
			Contains the length-adjusted time series (accounting for lags)

		scores : np.array
			Contains the scores for the time series
		"""

		Y = np.array(self.data[self.max_lag:self.data.shape[0]])
		scores = np.zeros(Y.shape[0])
		parm = np.array([self._param_desc[k]['prior'].transform(beta[k]) for k in range(beta.shape[0])])
		theta = np.ones(Y.shape[0])*parm[0]

		# Check if model has scale parameter
		if self.scale is True:
			if self.dist == 't':
				model_v = parm[parm.shape[0]-2]	
			else:
				model_v = 0		
			model_scale = parm[parm.shape[0]-1]
		else:
			model_scale = 0
			model_v = 0

		# Loop over time series
		for t in range(0,Y.shape[0]):

			if t < self.max_lag:

				theta[t] = parm[0]/(1-np.sum(parm[1:(self.ar+1)]))
			else:

				# Loop over AR terms
				for ar_term in range(0,self.ar):
					theta[t] += parm[1+ar_term]*theta[t-ar_term-1]

				# Loop over Score terms
				for sc_term in range(0,self.sc):
					theta[t] += parm[1+self.ar+sc_term]*scores[t-sc_term-1]

			# Calculate scores
			scores[t] = lik_score(Y[t],self.link(theta[t]),model_scale,model_v,self.dist)

		return theta, Y, scores

	def _mean_prediction(self,theta,Y,scores,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		theta : np.array
			The past predicted values

		Y : np.array
			The past data

		scores : np.array
			The past scores

		h : int
			How many steps ahead for the prediction

		t_params : np.array
			A vector of (transformed) parameters

		Returns
		----------
		h-length vector of mean predictions
		"""		

		# Create arrays to iteratre over
		Y_exp = Y.copy()
		theta_exp = theta.copy()
		scores_exp = scores.copy()

		# Loop over h time periods			
		for t in range(0,h):
			new_value = t_params[0]

			if self.ar != 0:
				for j in range(1,self.ar+1):
					new_value += t_params[j]*theta_exp[theta_exp.shape[0]-j]

			if self.sc != 0:
				for k in range(1,self.sc+1):
					new_value += t_params[k+self.ar]*scores_exp[scores_exp.shape[0]-k]

			Y_exp = np.append(Y_exp,[self.link(new_value)])
			theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
			scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

		return Y_exp

	def _sim_prediction(self,theta,Y,scores,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		theta : np.array
			The past predicted values

		Y : np.array
			The past data

		scores : np.array
			The past scores

		h : int
			How many steps ahead for the prediction

		t_params : np.array
			A vector of (transformed) parameters

		simulations : int
			How many simulations to perform

		Returns
		----------
		Matrix of simulations
		"""		

		sim_vector = np.zeros([simulations,h])

		for n in range(0,simulations):
			# Create arrays to iteratre over		
			Y_exp = Y.copy()
			theta_exp = theta.copy()
			scores_exp = scores.copy()

			# Loop over h time periods			
			for t in range(0,h):
				new_value = t_params[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += t_params[j]*theta_exp[theta_exp.shape[0]-j]

				if self.sc != 0:
					for k in range(1,self.sc+1):
						new_value += t_params[k+self.ar]*scores_exp[scores_exp.shape[0]-k]

				if self.dist == "Normal":
					rnd_value = np.random.normal(new_value, t_params[t_params.shape[0]-1], 1)[0]
				elif self.dist == "Laplace":
					rnd_value = np.random.laplace(new_value, t_params[t_params.shape[0]-1], 1)[0]
				elif self.dist == "Poisson":
					rnd_value = np.random.poisson(self.link(new_value), 1)[0]
				elif self.dist == "Exponential":
					rnd_value = np.random.exponential(1/self.link(new_value), 1)[0]
				elif self.dist == 't':
					rnd_value = new_value + t_params[t_params.shape[0]-1]*np.random.standard_t(t_params[t_params.shape[0]-2],1)[0]

				Y_exp = np.append(Y_exp,[rnd_value])
				theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
				scores_exp = np.append(scores_exp,scores[np.random.randint(scores.shape[0])]) # expectation of score is zero

			sim_vector[n] = Y_exp[(Y_exp.shape[0]-h):Y_exp.shape[0]]

		return np.transpose(sim_vector)

	def _summarize_simulations(self,mean_values,sim_vector,date_index,h,past_values):
		""" Summarizes a simulation vector and a mean vector of predictions

		Parameters
		----------
		mean_values : np.array
			Mean predictions for h-step ahead forecasts

		sim_vector : np.array
			N simulation predictions for h-step ahead forecasts

		date_index : pd.DateIndex or np.array
			Dates for the simulations

		h : int
			How many steps ahead are forecast

		past_values : int
			How many past observations to include in the forecast plot

		intervals : Boolean
			Would you like to show prediction intervals for the forecast?
		"""	

		error_bars = []
		for pre in range(5,100,5):
			error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[(mean_values.shape[0]-h):(mean_values.shape[0])],0,0))
		forecasted_values = mean_values[(mean_values.shape[0]-h-1):(mean_values.shape[0])]
		plot_values = mean_values[mean_values.shape[0]-h-past_values:mean_values.shape[0]]
		plot_index = date_index[len(date_index)-h-past_values:len(date_index)]
		return error_bars, forecasted_values, plot_values, plot_index

	def likelihood(self,beta):
		""" Creates the negative log-likelihood of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		The negative logliklihood of the model
		"""		

		theta, Y, scores = self._model(beta)

		if self.dist == "Laplace":
			return -np.sum(ss.laplace.logpdf(Y,loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))
		elif self.dist == "Normal":
			return -np.sum(ss.norm.logpdf(Y,loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))	
		elif self.dist == "Poisson":
			return -np.sum(ss.poisson.logpmf(Y,self.link(theta)))
		elif self.dist == "Exponential":
			return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))
		elif self.dist == "t":
			return -np.sum(ss.t.logpdf(x=Y,df=self._param_desc[beta.shape[0]-2]['prior'].transform(beta[beta.shape[0]-2]),loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))

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
			plt.figure(figsize=figsize)
			date_index = self.index[max(self.ar,self.sc):self.data.shape[0]]
			mu, Y, scores = self._model(self.params)

			plt.plot(date_index,Y,label='Data')
			plt.plot(date_index,self.link(mu),label='Filter',c='black')
			plt.title(self.data_name)
			plt.legend(loc=2)	
			plt.show()				
	
	def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):
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
		"""		

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Retrieve data, dates and (transformed) parameters
			theta, Y, scores = self._model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self._mean_prediction(theta,Y,scores,h,t_params)
			sim_values = self._sim_prediction(theta,Y,scores,h,t_params,15000)
			error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

			plt.figure(figsize=figsize)
			if intervals == True:
				alpha =[0.15*i/float(100) for i in range(50,12,-2)]
				for count, pre in enumerate(error_bars):
					plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])			
			
			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()

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
			x = GAS(ar=self.ar,sc=self.sc,dist=self.dist,integ=self.integ,data=self.data_original[0:(self.data_original.shape[0]-h+t)])
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

	def predict(self,h=5):
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		Returns
		----------
		- pd.DataFrame with predicted values
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			theta, Y, scores = self._model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			mean_values = self._mean_prediction(theta,Y,scores,h,t_params)
			forecasted_values = mean_values[(mean_values.shape[0]-h):mean_values.shape[0]]
			result = pd.DataFrame(forecasted_values)
			result.rename(columns={0:self.data_name}, inplace=True)
			result.index = date_index[(len(date_index)-h):len(date_index)]

			return result