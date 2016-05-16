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
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import gas as gas
from .. import data_check as dc

class EGARCH(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** BETA-t-EGARCH MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.array
		Field to specify the time series data that will be used.

	p : int
		Field to specify how many GARCH terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		fitting methods (MLE/MAP).

	q : int
		Field to specify how many SCORE terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		fitting methods(MLE/MAP).

	target : str (pd.DataFrame) or int (np.array)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,p,q,target=None):

		# Initialize TSM object
		super(EGARCH,self).__init__('EGARCH')

		# Parameters
		self.p = p
		self.q = q
		self.param_no = self.p + self.q + 2
		self.max_lag = max(self.p,self.q)
		self.leverage = False
		self.model_name = "EGARCH(" + str(self.p) + "," + str(self.q) + ")"
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

		# Format the data
		self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)

		self._param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None), 'q': dst.q_Normal(0,3)})		
		
		# GARCH terms
		for j in range(1,self.p+1):
			self._param_desc.append({'name' : 'p(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.q_Normal(0,3)})
		
		# SCORE terms
		for k in range(self.p+1,self.p+self.q+1):
			self._param_desc.append({'name' : 'q(' + str(k-self.q) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.q_Normal(0,3)})

		# For t-distribution
		self._param_desc.append({'name' : 'v', 'index': self.q+self.p+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		# Starting Parameters for Estimation
		self.starting_params = np.zeros(self.param_no)
		self.starting_params[0] = self._param_desc[0]['prior'].itransform(np.log(np.mean(np.power(self.data,2))))

	def _model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		lambda : np.array
			Contains the values for the conditional volatility series

		Y : np.array
			Contains the length-adjusted time series (accounting for lags)

		scores : np.array
			Contains the score terms for the time series
		"""

		Y = np.array(self.data[self.max_lag:self.data.shape[0]])
		X = np.ones(Y.shape[0])
		scores = np.zeros(Y.shape[0])

		# Transform parameters
		parm = np.array([self._param_desc[k]['prior'].transform(beta[k]) for k in range(beta.shape[0])])

		lmda = np.ones(Y.shape[0])*parm[0]

		# Loop over time series
		for t in range(0,Y.shape[0]):

			if t < self.max_lag:

				lmda[t] = parm[0]/(1-np.sum(parm[1:(self.p+1)]))

			else:

				# Loop over GARCH terms
				for p_term in range(0,self.p):
					lmda[t] += parm[1+p_term]*lmda[t-p_term-1]

				# Loop over Score terms
				for q_term in range(0,self.q):
					lmda[t] += parm[1+self.p+p_term]*scores[t-q_term-1]

				if self.leverage is True:
					lmda[t] += parm[1+self.p+self.q]*np.sign(-Y[t-1])*(scores[t-1]+1)

			scores[t] = gas.lik_score(Y[t],lmda[t],parm[len(parm)-1],0,'Beta-t')

		return lmda, Y, scores

	def _mean_prediction(self,lmda,Y,scores,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		lmda : np.array
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
		lmda_exp = lmda.copy()
		scores_exp = scores.copy()
		Y_exp = Y.copy()

		# Loop over h time periods			
		for t in range(0,h):
			new_value = t_params[0]

			if self.p != 0:
				for j in range(1,self.p+1):
					new_value += t_params[j]*lmda_exp[lmda_exp.shape[0]-j]

			if self.q != 0:
				for k in range(1,self.q+1):
					new_value += t_params[k+self.p]*scores_exp[scores_exp.shape[0]-k]

			if self.leverage is True:
				new_value += t_params[1+self.p+self.q]*np.sign(-Y_exp[scores_exp.shape[0]-1])*(scores_exp[scores_exp.shape[0]-1]+1)

			lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
			scores_exp = np.append(scores_exp,[0]) # expectation of score is zero
			Y_exp = np.append(Y_exp,[0])

		return lmda_exp

	def _sim_prediction(self,lmda,Y,scores,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		lmda : np.array
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
			lmda_exp = lmda.copy()
			scores_exp = scores.copy()
			Y_exp = Y.copy()

			# Loop over h time periods			
			for t in range(0,h):
				new_value = t_params[0]

				if self.p != 0:
					for j in range(1,self.p+1):
						new_value += t_params[j]*lmda_exp[lmda_exp.shape[0]-j]

				if self.q != 0:
					for k in range(1,self.q+1):
						new_value += t_params[k+self.p]*scores_exp[scores_exp.shape[0]-k]

				if self.leverage is True:
					new_value += t_params[1+self.p+self.q]*np.sign(-Y_exp[scores_exp.shape[0]-1])*(scores_exp[scores_exp.shape[0]-1]+1)

				lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
				scores_exp = np.append(scores_exp,scores[np.random.randint(scores.shape[0])]) # expectation of score is zero
				Y_exp = np.append(Y_exp,Y[np.random.randint(Y.shape[0])]) # bootstrap returns

			sim_vector[n] = lmda_exp[(lmda_exp.shape[0]-h):lmda_exp.shape[0]]

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

	def add_leverage(self):
		self.leverage = True
		self.param_no += 1
		self._param_desc.pop()
		self._param_desc.append({'name' : 'Leverage term', 'index': self.q+self.p+1, 'prior': ifr.Uniform(transform=None), 'q': dst.q_Normal(0,3)})		
		self._param_desc.append({'name' : 'v', 'index': self.q+self.p+2, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})
		self.starting_params = np.zeros(self.param_no)
		self.starting_params[0] = self._param_desc[0]['prior'].itransform(np.log(np.mean(np.power(self.data,2))))

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

		lmda, Y, ___ = self._model(beta)
		return -np.sum(ss.t.logpdf(x=Y,df=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1]),loc=np.zeros(lmda.shape[0]),scale=np.exp(lmda/float(2))))
	
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
			date_index = self.index[max(self.p,self.q):self.data.shape[0]]
			sigma2, Y, ___ = self._model(self.params)
			plt.plot(date_index,np.abs(Y),label=self.data_name + ' Absolute Values')
			plt.plot(date_index,np.exp(sigma2/2.0),label='EGARCH(' + str(self.p) + ',' + str(self.q) + ') std',c='black')					
			plt.title(self.data_name + " Volatility Plot")	
			plt.legend(loc=2)	
			plt.show()				

	def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):

		figsize = kwargs.get('figsize',(10,7))
		
		""" Plots forecast with the estimated model

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

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Retrieve data, dates and (transformed) parameters
			lmda, Y, scores = self._model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self._mean_prediction(lmda,Y,scores,h,t_params)
			sim_values = self._sim_prediction(lmda,Y,scores,h,t_params,15000)
			error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

			plt.figure(figsize=figsize)
			if intervals == True:
				alpha =[0.15*i/float(100) for i in range(50,12,-2)]
				for count, pre in enumerate(error_bars):
					plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], np.exp((forecasted_values-pre)/2), np.exp((forecasted_values+pre)/2),alpha=alpha[count])			
			
			plt.plot(plot_index,np.exp(plot_values/2.0))
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
			x = EGARCH(p=self.p,q=self.q,data=self.data[0:(self.data.shape[0]-h+t)])
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
		date_index = self.index[(len(self.index)-h):len(self.index)]
		predictions = self.predict_is(h)
		data = self.data[(len(self.index)-h):len(self.index)]

		plt.plot(date_index,np.abs(data),label='Data')
		plt.plot(date_index,predictions,label='Predictions',c='black')
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

			sigma2, Y, scores = self._model(self.params)	
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			mean_values = self._mean_prediction(sigma2,Y,scores,h,t_params)
			forecasted_values = mean_values[(mean_values.shape[0]-h):mean_values.shape[0]]
			result = pd.DataFrame(np.exp(forecasted_values/2.0))
			result.rename(columns={0:self.data_name}, inplace=True)
			result.index = date_index[(len(date_index)-h):len(date_index)]

			return result


