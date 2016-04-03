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
import numdifftools as nd

class GARCH(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSCEDASTICITY (GARCH) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	p : int
		Field to specify how many GARCH terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	q : int
		Field to specify how many ARCH terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,p,q,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'GARCH')

		# Parameters
		self.p = p
		self.q = q
		self.param_no = self.p + self.q + 1
		self.max_lag = max(self.p,self.q)
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

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

		self.param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform='exp'), 'q': dst.Normal(0,3)})		
		
		# ARCH terms e^2 (q)
		for j in range(1,self.q+1):
			self.param_desc.append({'name' : 'q(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# GARCH terms sigma (p)
		for k in range(self.q+1,self.p+self.q+1):
			self.param_desc.append({'name' : 'p(' + str(k-self.q) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})


	def model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		sigma2 : np.ndarray
			Contains the values for the conditional volatility series

		Y : np.ndarray
			Contains the length-adjusted time series (accounting for lags)

		eps : np.ndarray
			Contains the squared residuals (ARCH terms) for the time series
		"""

		xeps = np.power(self.data,2)
		Y = np.array(self.data[self.max_lag:len(self.data)])
		eps = np.power(Y,2)
		X = np.ones(len(Y))

		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# ARCH terms
		if self.q != 0:
			for i in range(self.q):	
				X = np.vstack((X,xeps[(self.max_lag-i-1):(len(xeps)-i-1)]))
			sigma2 = np.matmul(np.transpose(X),parm[0:len(parm)-self.p])
		else:
			sigma2 = np.transpose(X*parm[0])

		# GARCH terms
		if self.p != 0:
			for t in range(len(Y)):
				if t < self.max_lag:
					sigma2[t] = parm[0]/(1-np.sum(parm[(self.q+1):(self.q+self.p+1)]))
				elif t >= self.max_lag:
					for k in range(self.p):
						sigma2[t] += parm[1+self.q+k]*(sigma2[t-1-k])

		return sigma2, Y, eps

	# Returns negative log likelihood
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

		sigma2, Y, __ = self.model(beta)
		return -np.sum(ss.norm.logpdf(Y,loc=np.zeros(len(sigma2)),scale=np.power(sigma2,0.5)))

	def mean_prediction(self,sigma2,Y,scores,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		sigma2 : np.ndarray
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
		sigma2_exp = copy.deepcopy(sigma2)
		scores_exp = copy.deepcopy(scores)

		# Loop over h time periods			
		for t in range(h):
			new_value = t_params[0]

			if self.q != 0:
				for j in range(1,self.q+1):
					new_value += t_params[j]*sigma2_exp[len(sigma2_exp)-j]

			if self.p != 0:
				for k in range(1,self.p+1):
					new_value += t_params[k+self.q]*scores_exp[len(scores_exp)-k]

			sigma2_exp = np.append(sigma2_exp,[new_value]) # For indexing consistency
			scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

		return sigma2_exp

	def sim_prediction(self,sigma2,Y,scores,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		sigma2 : np.ndarray
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
			sigma2_exp = copy.deepcopy(sigma2)
			scores_exp = copy.deepcopy(scores)

			# Loop over h time periods			
			for t in range(h):
				new_value = t_params[0]

				if self.q != 0:
					for j in range(1,self.q+1):
						new_value += t_params[j]*sigma2_exp[len(sigma2_exp)-j]

				if self.p != 0:
					for k in range(1,self.p+1):
						new_value += t_params[k+self.q]*scores_exp[len(scores_exp)-k]

				sigma2_exp = np.append(sigma2_exp,[new_value]) # For indexing consistency
				scores_exp = np.append(scores_exp,scores[np.random.randint(len(scores))]) # expectation of score is zero

			sim_vector[n] = sigma2_exp[(len(sigma2_exp)-h):(len(sigma2_exp))]

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
			sigma2, Y, scores = self.model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self.mean_prediction(sigma2,Y,scores,h,t_params)
			sim_values = self.sim_prediction(sigma2,Y,scores,h,t_params,15000)
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
