from math import exp, sqrt, log, tanh, pi
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
from .. import covariances as cov

from .kalman import *
from .llm import *

class NLLEV(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** NON-GAUSSIAN LOCAL LEVEL MODEL ****

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
		super(NLLEV,self).__init__('NLLEV')

		# Parameters
		self.integ = integ
		self.max_lag = 0
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"
		self.state_no = 1

		# Format the data
		self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
		self.data_original = self.data

		# Difference data
		X = self.data
		for order in range(self.integ):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		
		self.data_length = X
		self.cutoff = 0

		# Add parameter information

		self._param_desc.append({'name' : 'Sigma^2 level','index': 1, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

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
		T, Z, R, Q : np.array
			State space matrices used in KFS algorithm
		"""		

		T = np.identity(1)
		R = np.identity(1)
		Z = np.identity(1)
		Q = np.identity(1)*self._param_desc[0]['prior'].transform(beta[0])

		return T, Z, R, Q

	def _poisson_approximating_model(self,beta,T,Z,R,Q):
		""" Creates approximating Gaussian model for Poisson measurement density

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		T, Z, R, Q : np.array
			State space matrices used in KFS algorithm

		Returns
		----------

		H : np.array
			Approximating measurement variance matrix

		mu : np.array
			Approximating measurement constants
		"""		

		if hasattr(self, 'H'):
			H = self.H
		else:
			H = np.ones(self.data.shape[0])

		if hasattr(self, 'mu'):
			mu = self.mu
		else:
			mu = np.zeros(self.data.shape[0])

		alpha = np.array([np.zeros(self.data.shape[0])])
		tol = 100.0
		it = 0
		while tol > 10**-7 and it < 5:
			old_alpha = alpha[0]
			alpha, V = nl_univariate_KFS(self.data,Z,H,T,Q,R,mu)
			H = np.exp(-alpha[0])
			mu = self.data - alpha[0] - np.exp(-alpha[0])*(self.data - np.exp(alpha[0]))
			tol = np.mean(np.abs(alpha[0]-old_alpha))
			it += 1

		return H, mu

	def _t_approximating_model(self,beta,T,Z,R,Q):
		""" Creates approximating Gaussian model for t measurement density

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		T, Z, R, Q : np.array
			State space matrices used in KFS algorithm

		Returns
		----------

		H : np.array
			Approximating measurement variance matrix

		mu : np.array
			Approximating measurement constants
		"""		

		H = np.ones(self.data.shape[0])*self._param_desc[1]['prior'].transform(beta[1])
		mu = np.zeros(self.data.shape[0])

		return H, mu

	@classmethod
	def t(cls,data,integ=0,target=None):
		""" Creates t-distributed state space model

		Parameters
		----------
		data : np.array
			Contains the time series

		integ : int (default : 0)
			Specifies how many time to difference the time series.

		target : str (pd.DataFrame) or int (np.array)
			Specifies which column name or array index to use. By default, first
			column/array will be selected as the dependent variable.

		Returns
		----------
		- NLLEV.t object
		"""		

		x = NLLEV(data=data,integ=integ,target=target)
		x._param_desc.append({'name' : 'Sigma^2 irregular','index': 2, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})
		x._param_desc.append({'name' : 'v','index': 3, 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})
		x._approximating_model = x._t_approximating_model
		x.meas_likelihood = x.t_likelihood
		x.model_name = "t-distributed Local Level Model"
		x.param_no = 3	
		x.link = np.array
		temp = LLEV(data,integ=integ,target=target)
		temp.fit(printer=False)
		x.starting_params = np.array([temp.params[1],temp.params[0],2])
		x.starting_cov_matrix = np.diag(np.abs(x.starting_params)) 		
		return x

	@classmethod
	def Poisson(cls,data,integ=0,target=None):
		""" Creates Poisson-distributed state space model

		Parameters
		----------
		data : np.array
			Contains the time series

		integ : int (default : 0)
			Specifies how many time to difference the time series.

		target : str (pd.DataFrame) or int (np.array)
			Specifies which column name or array index to use. By default, first
			column/array will be selected as the dependent variable.

		Returns
		----------
		- NLLEV.Poisson object
		"""		

		x = NLLEV(data=data,integ=integ,target=target)
		x._approximating_model = x._poisson_approximating_model
		x.meas_likelihood = x.poisson_likelihood
		x.model_name = "Poisson Local Level Model"	
		x.param_no = 1	
		x.link = np.exp
		temp = LLEV(data,integ=integ,target=target)
		temp.fit(printer=False)
		x.starting_params = np.array([temp.params[1]])
		x.starting_cov_matrix = np.diag(np.abs(x.starting_params)) 
		return x

	def fit(self,nsims=1000,smoother_weight=0.1,printer=True,*args,**kwargs):
		""" Fits model using Metropolis-Hastings

		Parameters
		----------
		nsims : int
			Number of simulations to perform (one long chain)

		smoother weight : float
			Specifies how much weight to give to simulation smoother sampler
			as opposed to the current states. E.g. 0.1*sample + 0.9*old_state.
			This is a tuning parameter that the user should tune manually to
			get reasonable acceptance rates.

		printer : Boolean
			Whether to print output or not

		Returns
		----------
		- self.params - parameters
		- self.states, self.states_mean, self.states_median - state information
		- self.states_upper_95, self.states_lower_95 - state credibility intervals
		"""		

		figsize = kwargs.get('figsize',(15,15))

		scale = 2.32/sqrt(self.param_no+self.data.shape[0])
		sampler = ifr.MetropolisHastings(self.posterior,scale,nsims,self.starting_params,cov_matrix=self.starting_cov_matrix,model_object=self)
		chain, mean_est, median_est, upper_95_est, lower_95_est, states, states_mean, states_median, states_upper_95, states_lower_95 = sampler.spdk_sample(smoother_weight=smoother_weight)	

		self.params = np.asarray(mean_est)

		if len(self._param_desc) == 1:
			chain = np.array([self._param_desc[0]['prior'].transform(chain)])
			mean_est = np.array([self._param_desc[0]['prior'].transform(mean_est)])
			median_est = np.array([self._param_desc[0]['prior'].transform(median_est)])
			upper_95_est = np.array([self._param_desc[0]['prior'].transform(upper_95_est)])
			lower_95_est = np.array([self._param_desc[0]['prior'].transform(lower_95_est)])
		else:

			for k in range(len(chain)):
				chain[k] = self._param_desc[k]['prior'].transform(chain[k])
				mean_est[k] = self._param_desc[k]['prior'].transform(mean_est[k])
				median_est[k] = self._param_desc[k]['prior'].transform(median_est[k])
				upper_95_est[k] = self._param_desc[k]['prior'].transform(upper_95_est[k])
				lower_95_est[k] = self._param_desc[k]['prior'].transform(lower_95_est[k])	

		self.chains = chain

		if printer == True:

			data = []

			for i in range(len(self._param_desc)):
				data.append({'param_name': self._param_desc[i]['name'], 'param_mean':np.round(mean_est[i],4), 'param_median':np.round(median_est[i],4), 'ci': "(" + str(np.round(lower_95_est[i],4)) + " | " + str(np.round(upper_95_est[i],4)) + ")"})

			fmt = [
				('Parameter','param_name',20),
				('Median','param_median',10),
				('Mean', 'param_mean', 15),
				('95% Credibility Interval','ci',25)]


			print(self.model_name)
			print("==================")
			print("Method: Metropolis-Hastings")
			print("Number of simulations: " + str(nsims))
			print("Number of observations: " + str(len(self.data)-self.cutoff))
			print("Unnormalized Log Posterior: " + str(np.round(-self.posterior(self.params,states),4)))
			print("")
			print( op.TablePrinter(fmt, ul='=')(data) )

		fig = plt.figure(figsize=figsize)

		# Loop over evolution parameters
		for j in range(len(self.params)):

			for k in range(4):
				iteration = j*4 + k + 1
				ax = fig.add_subplot(len(self.params)+states.shape[1],4,iteration)

				if iteration in range(1,len(self.params)*4 + 1,4):
					a = sns.distplot(chain[j], rug=False, hist=False)
					a.set_ylabel(self._param_desc[j]['name'])
					if iteration == 1:
						a.set_title('Density Estimate')
				elif iteration in range(2,len(self.params)*4 + 1,4):
					a = plt.plot(chain[j])
					if iteration == 2:
						plt.title('Trace Plot')
				elif iteration in range(3,len(self.params)*4 + 1,4): 
					plt.plot(np.cumsum(chain[j])/np.array(range(1,len(chain[j])+1)))
					if iteration == 3:
						plt.title('Cumulative Average')					
				elif iteration in range(4,len(self.params)*4 + 1,4):
					plt.bar(range(1,10),[cov.acf(chain[j],lag) for lag in range(1,10)])
					if iteration == 4:
						plt.title('ACF Plot')	

		# Plot out traceplots of the trace
		for j in range(states.shape[1]):

			for k in range(4):
				iteration = len(self.params)*4 + j*4 + k + 1
				ax = fig.add_subplot(len(self.params)+states.shape[1],4,iteration)

				if iteration in range(1,(len(self.params)+states.shape[1])*4 + 1,4):
					for parameter in range(states.shape[2]):
						a = sns.distplot(states[:,j,parameter].T, rug=False, hist=False)
						a.set_ylabel('States ' + str(j+1))
				elif iteration in range(2,(len(self.params)+states.shape[1])*4 + 1,4):
					a = plt.plot(states[:,j,:])
				elif iteration in range(3,(len(self.params)+states.shape[1])*4 + 1,4): 
					for parameter in range(states.shape[2]):
						plt.plot(np.cumsum(states[:,j,parameter])/np.array(range(1,len(states[:,j,parameter])+1)))		
				elif iteration in range(4,(len(self.params)+states.shape[1])*4 + 1,4):
					plt.bar(range(1,10),[cov.acf(states[:,j,0],lag) for lag in range(1,10)])

		sns.plt.show()		

		self.states = states
		self.states_mean = states_mean
		self.states_median = states_median
		self.states_upper_95 = states_upper_95
		self.states_lower_95 = states_lower_95

	def posterior(self,beta,alpha):
		""" Returns negative log posterior

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		alpha : np.array
			State matrix

		Returns
		----------
		Negative log posterior
		"""

		post = self.likelihood(beta,alpha)
		for k in range(0,self.param_no):
			post += -self._param_desc[k]['prior'].logpdf(beta[k])
		return post		

	def state_likelihood(self,beta,alpha):
		""" Returns likelihood of the states given the evolution parameters

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		alpha : np.array
			State matrix

		Returns
		----------
		State likelihood
		"""

		_, _, _, Q = self._ss_matrices(beta)
		residuals = alpha[0][1:alpha[0].shape[0]]-alpha[0][0:alpha[0].shape[0]-1]
		return np.sum(ss.norm.logpdf(residuals,loc=0,scale=np.power(Q.ravel(),0.5)))

	def likelihood(self,beta,alpha):
		""" Creates negative loglikelihood of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		alpha : np.array
			A vector of states

		Returns
		----------
		Negative loglikelihood
		"""		

		return -(self.state_likelihood(beta,alpha) + self.meas_likelihood(beta,alpha))

	def poisson_likelihood(self,beta,alpha):
		""" Creates Poisson loglikelihood of the data given the states

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		alpha : np.array
			A vector of states

		Returns
		----------
		Poisson loglikelihood
		"""		

		return np.sum(ss.poisson.logpmf(self.data,np.exp(alpha[0])))

	def t_likelihood(self,beta,alpha):
		""" Creates t loglikelihood of the date given the states

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		alpha : np.array
			A vector of states

		Returns
		----------
		t loglikelihood
		"""		

		return np.sum(ss.t.logpdf(x=self.data,df=self._param_desc[2]['prior'].transform(beta[2]),loc=alpha[0],scale=self._param_desc[1]['prior'].transform(beta[1])))

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
			previous_value = self.states_mean[self.states_mean.shape[0]-2]	
			forecasted_values = np.ones(h)*self.states_mean[self.states_mean.shape[0]-1]	
			date_index = self.shift_dates(h)
			simulations = 10000
			sim_vector = np.zeros([simulations,h])

			for n in range(0,simulations):	
				rnd_q = np.random.normal(0,np.sqrt(self._param_desc[0]['prior'].transform(self.params[0])),h)	
				exp = forecasted_values.copy()

				for t in range(0,h):
					if t == 0:
						exp[t] = forecasted_values[t] + rnd_q[t]
					else:
						exp[t] = exp[t-1] + rnd_q[t]

				sim_vector[n] = exp

			sim_vector = self.link(np.transpose(sim_vector))
			forecasted_values = self.link(forecasted_values)
			previous_value = self.link(previous_value)

			plt.figure(figsize=figsize)	

			error_bars = []
			for pre in range(5,100,5):
				error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - forecasted_values,0,0))

			if intervals == True:
				alpha =[0.15*i/float(100) for i in range(50,12,-2)]
				for count, pre in enumerate(error_bars):
					plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], np.insert(forecasted_values,0,previous_value)-pre, np.insert(forecasted_values,0,previous_value)+pre,alpha=alpha[count])			

			plot_values = np.append((self.link(self.states_mean[self.states_mean.shape[0]-1-past_values:self.states_mean.shape[0]-1])),forecasted_values)
			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

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

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			date_index = copy.deepcopy(self.index)
			date_index = date_index[self.integ:self.data_original.shape[0]+1]
			
			plt.figure(figsize=figsize)	
			
			plt.subplot(2, 1, 1)
			plt.title(self.data_name + " Raw and Smoothed")	
			plt.plot(date_index,self.data,label='Data')
			plt.plot(date_index,self.link(self.states_mean),label='Smoothed',c='black')
			plt.legend(loc=2)
			
			plt.subplot(2, 1, 2)
			plt.title(self.data_name + " Local Level")	
			plt.plot(date_index,self.link(self.states_mean),label='Smoothed State')
			plt.plot(date_index,self.link(self.states_upper_95),label='95% Credible Interval', c='black',alpha=0.2)
			plt.plot(date_index,self.link(self.states_lower_95),c='black',alpha=0.2)
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
		- pd.DataFrame with predictions
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:
			# Retrieve data, dates and (transformed) parameters			
			date_index = self.shift_dates(h)
			forecasted_values = np.ones(h)*self.states_mean[self.states_mean.shape[0]-1]

			result = pd.DataFrame(self.link(forecasted_values))
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
			x = NLLEV(integ=self.integ,data=self.data_original[0:(self.data_original.shape[0]-h+t)])
			x.fit(printer=False,nsims=100)
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

		plt.plot(date_index,data,label='Data')
		plt.plot(date_index,predictions,label='Predictions',c='black')
		plt.title(self.data_name)
		plt.legend(loc=2)	
		plt.show()			

	def simulation_smoother(self,beta):
		""" Durbin and Koopman simulation smoother - simulates from states 
		given model parameters and observations

		Parameters
		----------

		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		- A simulated state evolution
		"""			

		T, Z, R, Q = self._ss_matrices(beta)
		H, mu = self._approximating_model(beta,T,Z,R,Q)

		# Generate e_t+ and n_t+
		rnd_h = np.random.normal(0,np.sqrt(H),self.data.shape[0])
		q_dist = ss.multivariate_normal([0.0], Q)
		rnd_q = q_dist.rvs(self.data.shape[0])

		# Generate a_t+ and y_t+
		a_plus = np.zeros((T.shape[0],self.data.shape[0])) 
		y_plus = np.zeros(self.data.shape[0])

		for t in range(0,self.data.shape[0]):
			if t == 0:
				a_plus[:,t] = np.dot(T,a_plus[:,t]) + rnd_q[t]
				y_plus[t] = mu[t] + np.dot(Z,a_plus[:,t]) + rnd_h[t]
			else:
				if t != self.data.shape[0]:
					a_plus[:,t] = np.dot(T,a_plus[:,t-1]) + rnd_q[t]
					y_plus[t] = mu[t] + np.dot(Z,a_plus[:,t]) + rnd_h[t]

		alpha_hat = self.smoothed_state(self.data,beta, H, mu)
		alpha_hat_plus = self.smoothed_state(y_plus,beta, H, mu)
		alpha_tilde = alpha_hat - alpha_hat_plus + a_plus
		
		return alpha_tilde

	def smoothed_state(self,data,beta, H, mu):
		""" Creates smoothed state estimate given state matrices and 
		parameters.

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

		T, Z, R, Q = self._ss_matrices(beta)
		alpha, V = nl_univariate_KFS(data,Z,H,T,Q,R,mu)
		return alpha