from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import gas as gas
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

class EGARCH(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** BETA-t-EGARCH MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.ndarray
		Field to specify the time series data that will be used.

	p : int
		Field to specify how many GARCH terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	q : int
		Field to specify how many SCORE terms the model will have. Warning:
		higher-order lag specifications often fail to return for optimization
		methods of inference (MLE/MAP).

	target : str (pd.DataFrame) or int (np.ndarray)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,p,q,target=None):

		# Initialize TSM object
		tsm.TSM.__init__(self,'EGARCH')

		# Parameters
		self.p = p
		self.q = q
		self.param_no = self.p + self.q + 2
		self.max_lag = max(self.p,self.q)
		self.hess_type = 'numerical'
		self.param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

		# Format the data
		self.data, self.data_name, self.data_type, self.index = dc.data_check(data,target)

		self.param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None), 'q': dst.Normal(0,3)})		
		
		# GARCH terms
		for j in range(1,self.p+1):
			self.param_desc.append({'name' : 'p(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})
		
		# SCORE terms
		for k in range(self.p+1,self.p+self.q+1):
			self.param_desc.append({'name' : 'q(' + str(k-self.q) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform=None), 'q': dst.Normal(0,3)})

		# For t-distribution
		self.param_desc.append({'name' : 'v', 'index': self.q+self.p+1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.Normal(0,3)})


	# Holds the core model matrices
	def model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		lambda : np.ndarray
			Contains the values for the conditional volatility series

		Y : np.ndarray
			Contains the length-adjusted time series (accounting for lags)

		scores : np.ndarray
			Contains the score terms for the time series
		"""

		Y = np.array(self.data[self.max_lag:len(self.data)])
		X = np.ones(len(Y))
 		scores = np.zeros(len(Y))

		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		lmda = np.ones(len(Y))*parm[0]

		# Loop over time series
		for t in range(len(Y)):

			if t < self.max_lag:

				lmda[t] = parm[0]/(1-np.sum(parm[1:(self.p+1)]))

			else:

				# Loop over GARCH terms
				for p_term in range(self.p):
					lmda[t] += parm[1+p_term]*lmda[t-p_term-1]

				# Loop over Score terms
				for q_term in range(self.q):
					lmda[t] += parm[1+self.p+p_term]*scores[t-q_term-1]

			scores[t] = gas.lik_score(Y[t],lmda[t],parm[len(parm)-1],'Beta-t')

		return lmda, Y, scores

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

		lmda, Y, ___ = self.model(beta)
		return -np.sum(ss.t.logpdf(x=Y,df=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1]),loc=np.zeros(len(lmda)),scale=np.exp(lmda/float(2))))
		
	def mean_prediction(self,lmda,Y,scores,h,t_params):
		""" Creates a h-step ahead mean prediction

		Parameters
		----------
		lmda : np.ndarray
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
		lmda_exp = copy.deepcopy(lmda)
		scores_exp = copy.deepcopy(scores)

		# Loop over h time periods			
		for t in range(h):
			new_value = t_params[0]

			if self.q != 0:
				for j in range(1,self.q+1):
					new_value += t_params[j]*lmda_exp[len(lmda_exp)-j]

			if self.p != 0:
				for k in range(1,self.p+1):
					new_value += t_params[k+self.q]*scores_exp[len(scores_exp)-k]

			lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
			scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

		return lmda_exp

	def sim_prediction(self,lmda,Y,scores,h,t_params,simulations):
		""" Simulates a h-step ahead mean prediction

		Parameters
		----------
		lmda : np.ndarray
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
			lmda_exp = copy.deepcopy(lmda)
			scores_exp = copy.deepcopy(scores)

			# Loop over h time periods			
			for t in range(h):
				new_value = t_params[0]

				if self.q != 0:
					for j in range(1,self.q+1):
						new_value += t_params[j]*lmda_exp[len(lmda_exp)-j]

				if self.p != 0:
					for k in range(1,self.p+1):
						new_value += t_params[k+self.q]*scores_exp[len(scores_exp)-k]

				lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
				scores_exp = np.append(scores_exp,scores[np.random.randint(len(scores))]) # expectation of score is zero

			sim_vector[n] = lmda_exp[(len(lmda_exp)-h):(len(lmda_exp))]

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
			lmda, Y, scores = self.model(self.params)			
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get mean prediction and simulations (for errors)
			mean_values = self.mean_prediction(lmda,Y,scores,h,t_params)
			sim_values = self.sim_prediction(lmda,Y,scores,h,t_params,15000)
			error_bars, forecasted_values, plot_values, plot_index = self.summarize_simulations(mean_values,sim_values,date_index,h,past_values)

			plt.figure(figsize=figsize)
			if intervals == True:
				alpha =[0.15*i/float(100) for i in range(50,12,-2)]
				for count, pre in enumerate(error_bars):
					plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], np.exp((forecasted_values-pre)/2), np.exp((forecasted_values+pre)/2),alpha=alpha[count])			
			
			plt.plot(plot_index,np.exp(plot_values/2))
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()

			self.predictions = {'error_bars' : error_bars, 'forecasted_values' : forecasted_values, 'plot_values' : plot_values, 'plot_index': plot_index}

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
			date_index = self.index[max(self.p,self.q):len(self.data)]
			sigma2, Y, ___ = self.model(self.params)
			plt.plot(date_index,np.abs(Y),label=self.data_name + ' Absolute Values')
			plt.plot(date_index,np.exp(sigma2/2),label='EGARCH(' + str(self.p) + ',' + str(self.q) + ') std',c='black')					
			plt.title(self.data_name + " Volatility Plot")	
			plt.legend(loc=2)	
			plt.show()				