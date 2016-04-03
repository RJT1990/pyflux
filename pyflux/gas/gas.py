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
import numdifftools as nd

class GAS(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	dist : str
		Field to specify the distribution. Options include 'Normal',
		'Laplace', 'Poisson' and 'Exponential'.

	ar : int
		Field to specify how many AR terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	sc : int
		Field to specify how many Score terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	integ : int (default : 0)
		Specifies how many time to difference the time series.

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""


	def __init__(self,data,dist,ar,sc,integ=0,target=None):
		tsm.TSM.__init__(self,'GAS')
		self.dist = dist
		self.ar = ar
		self.sc = sc
		self.integ = integ
		self.param_no = self.ar + self.sc + 1
		self.max_lag = max(self.ar,self.sc)
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
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
			raise ValueError("The data input is not pandas or numpy compatible!")

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
		
		for k in range(self.ar+1,self.ar+self.sc+1):
			self.param_desc.append({'name' : 'SC(' + str(k-self.ar) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		if self.scale is True:
			self.param_desc.append({'name' : 'Scale','index': self.ar+self.sc+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})


	def model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		theta : np.ndarray
			Contains the predicted values for the time series

		Y : np.ndarray
			Contains the length-adjusted time series (accounting for lags)

		scores : np.ndarray
			Contains the scores for the time series
		"""

		Y = np.array(self.data[self.max_lag:len(self.data)])
		scores = np.zeros(len(Y))
 		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]
		theta = np.ones(len(Y))*parm[0]

		# Check if model has scale parameter
		if self.scale is True:
			model_scale = parm[len(parm)-1]
		else:
			model_scale = 0

		# Loop over time series
		for t in range(len(Y)):

			if t < self.max_lag:

				theta[t] = parm[0]/(1-np.sum(parm[(1):(self.ar+1)]))
			else:

				# Loop over AR terms
				for ar_term in range(self.ar):
					theta[t] += parm[1+ar_term]*theta[t-ar_term-1]

				# Loop over Score terms
				for sc_term in range(self.sc):
					theta[t] += parm[1+self.ar+sc_term]*scores[t-sc_term-1]

			# Calculate scores
			scores[t] = lik_score(Y[t],self.link(theta[t]),model_scale,self.dist)

		return theta, Y, scores

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

		theta, Y, scores = self.model(beta)

		if self.dist == "Laplace":
			return -np.sum(ss.laplace.logpdf(Y,loc=theta,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))
		elif self.dist == "Normal":
			return -np.sum(ss.norm.logpdf(Y,loc=theta,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))	
		elif self.dist == "Poisson":
			return -np.sum(ss.poisson.logpmf(Y,self.link(theta)))
		elif self.dist == "Exponential":
			return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))
	
	def mean_prediction(self,theta,Y,scores,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		theta : np.ndarray
			The past predicted values

		Y : np.ndarray
			The past data

		scores : np.ndarray
			The past scores

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
		theta_exp = copy.deepcopy(theta)
		scores_exp = copy.deepcopy(scores)

		# Loop over h time periods			
		for t in range(h):
			new_value = t_params[0]

			if self.ar != 0:
				for j in range(1,self.ar+1):
					new_value += t_params[j]*theta_exp[len(theta_exp)-j]

			if self.sc != 0:
				for k in range(1,self.sc+1):
					new_value += t_params[k+self.ar]*scores_exp[len(scores_exp)-k]

			Y_exp = np.append(Y_exp,[self.link(new_value)])
			theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
			scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

		return Y_exp

	def sim_prediction(self,theta,Y,scores,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		theta : np.ndarray
			The past predicted values

		Y : np.ndarray
			The past data

		scores : np.ndarray
			The past scores

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
			theta_exp = copy.deepcopy(theta)
			scores_exp = copy.deepcopy(scores)

			# Loop over h time periods			
			for t in range(h):
				new_value = t_params[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += t_params[j]*theta_exp[len(theta_exp)-j]

				if self.sc != 0:
					for k in range(1,self.sc+1):
						new_value += t_params[k+self.ar]*scores_exp[len(scores_exp)-k]

				if self.dist == "Normal":
					rnd_value = np.random.normal(new_value, t_params[len(t_params)-1], 1)[0]
				elif self.dist == "Laplace":
					rnd_value = np.random.laplace(new_value, t_params[len(t_params)-1], 1)[0]
				elif self.dist == "Poisson":
					rnd_value = np.random.poisson(self.link(new_value), 1)[0]
				elif self.dist == "Exponential":
					rnd_value = np.random.exponential(1/self.link(new_value), 1)[0]

				Y_exp = np.append(Y_exp,[rnd_value])
				theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
				scores_exp = np.append(scores_exp,scores[np.random.randint(len(scores))]) # expectation of score is zero

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
			Would you like to show 90/95% prediction intervals for the forecast?
		"""	

		error_bars = []
		for pre in range(5,100,5):
			error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[(len(mean_values)-h):(len(mean_values))],0,0))
		forecasted_values = mean_values[(len(mean_values)-h-1):(len(mean_values))]
		plot_values = mean_values[len(mean_values)-h-past_values:len(mean_values)]
		plot_index = date_index[len(date_index)-h-past_values:len(date_index)]
		return error_bars, forecasted_values, plot_values, plot_index


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
			Would you like to show 90/95% prediction intervals for the forecast?

		Returns
		----------
		- Plot of the forecast
		- Error bars, forecasted_values, plot_values, plot_index
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Retrieve data, dates and (transformed) parameters
			theta, Y, scores = self.model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self.mean_prediction(theta,Y,scores,h,t_params)
			sim_values = self.sim_prediction(theta,Y,scores,h,t_params,15000)
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
