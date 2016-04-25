from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import exp, sqrt, log, tanh
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

class ARIMA(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** AUTOREGRESSIVE INTEGERATED MOVING AVERAGE (ARIMA) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	ar : int
		Field to specify how many AR terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	ma : int
		Field to specify how many MA terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,ar,ma,integ=0,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'ARIMA')

		# Parameters
		self.ar = ar
		self.ma = ma
		self.integ = integ
		self.param_no = self.ar + self.ma + 2
		self.max_lag = max(self.ar,self.ma)
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

		# Format the data
		self.data, self.data_name, self.data_type, self.index = dc.data_check(data,target)

		# Difference data
		X = self.data
		for order in range(self.integ):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		

		# Add parameter information

		self.param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None), 'q': dst.Normal(0,3)})		
		
		# AR priors
		for j in range(1,self.ar+1):
			self.param_desc.append({'name' : 'AR(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# MA priors
		for k in range(self.ar+1,self.ar+self.ma+1):
			self.param_desc.append({'name' : 'MA(' + str(k-self.ar) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# Variance prior
		self.param_desc.append({'name' : 'Sigma','index': self.ar+self.ma+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})


	def model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		mu : np.ndarray
			Contains the predicted values for the time series

		Y : np.ndarray
			Contains the length-adjusted time series (accounting for lags)
		"""		

		Y = np.array(self.data[self.max_lag:len(self.data)])
		X = np.ones(len(Y))

		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# AR terms
		if self.ar != 0:
			for i in range(self.ar):
				X = np.vstack((X,self.data[(self.max_lag-i-1):(len(self.data)-i-1)]))

		mu = np.matmul(np.transpose(X),parm[0:len(parm)-1-self.ma])

		# MA terms
		if self.ma != 0:
			for t in range(self.max_lag,len(Y)):
				for k in range(self.ma):
						mu[t] += parm[1+self.ar+k]*(Y[t-1-k]-mu[t-1-k])

		return mu, Y 

	def likelihood(self,beta):
		""" Creates the negative log-likelihood of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		The negative logliklihood of the model
		"""		

		mu, Y = self.model(beta)
		return -np.sum(ss.norm.logpdf(Y,loc=mu,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))

	def mean_prediction(self,mu,Y,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		mu : np.ndarray
			The past predicted values

		Y : np.ndarray
			The past data

		h : int
			How many steps ahead for the prediction

		t_params : np.ndarray
			A vector of (transformed) parameters

		Returns
		----------
		h-length vector of mean predictions
		"""		

		# Create arrays to iteratre over
		Y_exp = copy.deepcopy(Y)
		mu_exp = copy.deepcopy(mu)

		# Loop over h time periods			
		for t in range(h):
			new_value = t_params[0]

			if self.ar != 0:
				for j in range(1,self.ar+1):
					new_value += t_params[j]*Y_exp[len(Y_exp)-j]

			if self.ma != 0:
				for k in range(1,self.ma+1):
					if (k-1) >= t:
						new_value += t_params[k+self.ar]*(Y_exp[len(Y_exp)-k]-mu_exp[len(mu_exp)-k])

			Y_exp = np.append(Y_exp,[new_value])
			mu_exp = np.append(mu_exp,[0]) # For indexing consistency

		return Y_exp

	def sim_prediction(self,mu,Y,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		mu : np.ndarray
			The past predicted values

		Y : np.ndarray
			The past data

		h : int
			How many steps ahead for the prediction

		t_params : np.ndarray
			A vector of (transformed) parameters

		simulations : int
			How many simulations to perform

		Returns
		----------
		Matrix of simulations
		"""		

		sim_vector = np.zeros([simulations,h])

		for n in range(simulations):
			# Create arrays to iteratre over		
			Y_exp = copy.deepcopy(Y)
			mu_exp = copy.deepcopy(mu)

			# Loop over h time periods			
			for t in range(h):

				new_value = t_params[0] + np.random.randn(1)*t_params[len(t_params)-1]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += t_params[j]*Y_exp[len(Y_exp)-j]

				if self.ma != 0:
					for k in range(1,self.ma+1):
						if (k-1) >= t:
							new_value += t_params[k+self.ar]*(Y_exp[len(Y_exp)-k]-mu_exp[len(mu_exp)-k])

				Y_exp = np.append(Y_exp,[new_value])
				mu_exp = np.append(mu_exp,[0]) # For indexing consistency

				sim_vector[n] = Y_exp[(len(Y_exp)-h):(len(Y_exp))]

		return np.transpose(sim_vector)

	def summarize_simulations(self,mean_values,sim_vector,date_index,h,past_values):
		""" Summarizes a simulation vector and a mean vector of predictions

		Parameters
		----------
		mean_values : np.ndarray
			Mean predictions for h-step ahead forecasts

		sim_vector : np.ndarray
			N simulation predictions for h-step ahead forecasts

		date_index : pd.DateIndex or np.ndarray
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
			error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[(len(mean_values)-h):(len(mean_values))],0,0))
		forecasted_values = mean_values[(len(mean_values)-h-1):(len(mean_values))]
		plot_values = mean_values[len(mean_values)-h-past_values:len(mean_values)]
		plot_index = date_index[len(date_index)-h-past_values:len(date_index)]
		return error_bars, forecasted_values, plot_values, plot_index
		
	# Produces T-step ahead forecast for the series
	# This code is very inefficient; needs amending
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
			mu, Y = self.model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self.mean_prediction(mu,Y,h,t_params)
			sim_values = self.sim_prediction(mu,Y,h,t_params,15000)
			error_bars, forecasted_values, plot_values, plot_index = self.summarize_simulations(mean_values,sim_values,date_index,h,past_values)

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

			self.predictions = {'error_bars' : error_bars, 'forecasted_values' : forecasted_values, 'plot_values' : plot_values, 'plot_index': plot_index}





