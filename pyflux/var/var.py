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
import datetime

from .. import inference as ifr
from .. import distributions as dist
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

class VAR(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** VECTOR AUTOREGRESSION (VAR) MODELS ****

	Parameters
	----------
	data : pd.DataFrame or np.array
		Field to specify the time series data that will be used.

	lags : int
		Field to specify how many lag terms the model will have. 

	integ : int (default : 0)
		Specifies how many time to difference the dependent variables.

	target : str (pd.DataFrame) or int (np.array)
		Specifies which column name or array index to use. By default, first
		column/array will be selected as the dependent variable.
	"""

	def __init__(self,data,lags,target=None,integ=0):

		# Initialize TSM object		
		super(VAR,self).__init__('VAR')

		# Parameters
		self.lags = lags
		self.integ = integ
		self.max_lag = lags
		self.model_name = "VAR(" + str(self.lags) + ")"
		self._hess_type = 'numerical'
		self.supported_methods = ["OLS","MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "OLS"

		# Format the data
		self.data, self.data_name, self.is_pandas, self.index = dc.mv_data_check(data,target)
		self.data_original = self.data.copy()

		# Difference data
		X = np.transpose(self.data)
		for order in range(self.integ):
			X = np.asarray([np.diff(i) for i in X])
			self.data_name = np.asarray(["Differenced " + str(i) for i in self.data_name])
		self.data = X	
		self.ylen = len(self.data_name)

		# Create VAR parameters
		for variable in range(self.ylen):
			self._param_desc.append({'name' : self.data_name[variable] + ' Constant', 'index': len(self._param_desc), 'prior': ifr.Normal(0,3,transform=None), 'q': dist.q_Normal(0,3)})		
			other_variables = np.delete(range(self.ylen), [variable])
			for lag_no in range(self.lags):
				self._param_desc.append({'name' : str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', 'index': len(self._param_desc), 'prior': ifr.Normal(0,0.5,transform=None), 'q': dist.q_Normal(0,3)})
				for other in other_variables:
					self._param_desc.append({'name' : str(self.data_name[other]) + ' to ' + str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', 'index': len(self._param_desc), 'prior': ifr.Normal(0,0.5,transform=None), 'q': dist.q_Normal(0,3)})

		# Variance prior
		for i in range(self.ylen):
			for k in range(self.ylen):
				if i == k:
					self._param_desc.append({'name' : 'Sigma' + str(self.data_name[i]),'index': len(self._param_desc),'prior': ifr.Uniform(transform='exp'), 'q': dist.q_Normal(0,3)})
				elif i > k:
					self._param_desc.append({'name' : 'Sigma' + str(self.data_name[i]) + ' to ' + str(self.data_name[k]),'index': len(self._param_desc), 'prior': ifr.Uniform(transform=None), 'q': dist.q_Normal(0,3)})

		# Other attributes
		self._param_hide = len(self.data)**2 - (len(self.data)**2 - len(self.data))/2 # Whether to cutoff variance parameters from results		
		self.param_no = len(self._param_desc)

		# Starting parameters
		self.starting_params = self._create_B_direct().flatten()
		cov = self.ols_covariance()

		for i in range(0,self.ylen):
			for k in range(0,self.ylen):
				if i == k:
					self.starting_params = np.append(self.starting_params,self._param_desc[len(self.starting_params)]['prior'].itransform(cov[i,k]))
				elif i > k:
					self.starting_params = np.append(self.starting_params,self._param_desc[len(self.starting_params)]['prior'].itransform(cov[i,k]))			

	def _create_B(self,Y):
		""" Creates coefficient matrix

		Parameters
		----------
		Y : np.array
			The dependent variables Y

		Returns
		----------
		The coefficient matrix B
		"""			

		Z = self._create_Z(Y)
		return np.dot(np.dot(Y,np.transpose(Z)),np.linalg.inv(np.dot(Z,np.transpose(Z))))

	def _create_B_direct(self):
		""" Creates coefficient matrix (calculates Y within - for OLS fitting)

		Returns
		----------
		The coefficient matrix B
		"""			

		Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])		
		Z = self._create_Z(Y)
		return np.dot(np.dot(Y,np.transpose(Z)),np.linalg.inv(np.dot(Z,np.transpose(Z))))

	def _create_Z(self,Y):
		""" Creates design matrix holding the lagged variables

		Parameters
		----------
		Y : np.array
			The dependent variables Y

		Returns
		----------
		The design matrix Z
		"""	

		Z = np.ones(((self.ylen*self.lags +1),Y[0].shape[0]))
		row_count = 1
		for lag in range(1,self.lags+1):
			for reg in range(Y.shape[0]):
				Z[row_count,:] = self.data[reg][(self.lags-lag):self.data[reg].shape[0]-lag]			
				row_count += 1
		return Z

	def _forecast_mean(self,h,t_params,Y,shock_type=None,shock_index=0,shock_value=None,shock_dir='positive',irf_intervals=False):
		""" Function allows for mean prediction; also allows shock specification for simulations or impulse response effects

		Parameters
		----------
		h : int
			How many steps ahead to forecast

		t_params : np.array
			Transformed parameter vector

		Y : np.array
			Data for series that is being forecast

		shock_type : None or str
			Type of shock; options include None, 'Cov' (simulate from covariance matrix), 'IRF' (impulse response shock)

		shock_index : int
			Which parameter to apply the shock to if using an IRF.

		shock_value : None or float
			If specified, applies a custom-sized impulse response shock.

		shock_dir : str
			Direction of the IRF shock. One of 'positive' or 'negative'.

		irf_intervals : Boolean
			Whether to have intervals for the IRF plot or not

		Returns
		----------
		A vector of forecasted data
		"""			

		random = self._shock_create(h, shock_type, shock_index, shock_value, shock_dir,irf_intervals)
		exp = [Y[variable] for variable in range(0,self.ylen)]
		
		# Each forward projection
		for t in range(0,h):
			new_values = np.zeros(self.ylen)

			# Each variable
			for variable in range(0,self.ylen):
				index_ref = variable*(1+self.ylen*self.lags)
				new_values[variable] = t_params[index_ref] # constant

				# VAR(p) terms
				for lag in range(0,self.lags):
					for lagged_var in range(0,self.ylen):
						new_values[variable] += t_params[index_ref+lagged_var+(lag*self.ylen)+1]*exp[lagged_var][exp[lagged_var].shape[0]-1-lag]
				
				# Random shock
				new_values[variable] += random[t][variable]

			# Add new values
			for variable in range(0,self.ylen):
				exp[variable] = np.append(exp[variable],new_values[variable])

		return np.array(exp)

	def _model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		mu : np.array
			Contains the predicted values for the time series

		Y : np.array
			Contains the length-adjusted time series (accounting for lags)
		"""		

		Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])

		# Transform parameters
		beta = np.array([self._param_desc[k]['prior'].transform(beta[k]) for k in range(beta.shape[0])])

		params = []
		col_length = 1 + self.ylen*self.lags
		for i in range(0,self.ylen):
			params.append(beta[(col_length*i): (col_length*(i+1))])

		mu = np.dot(np.array(params),self._create_Z(Y))

		return mu, Y

	def _shock_create(self, h, shock_type, shock_index, shock_value, shock_dir, irf_intervals):
		""" Function creates shocks based on desired specification

		Parameters
		----------
		h : int
			How many steps ahead to forecast

		shock_type : None or str
			Type of shock; options include None, 'Cov' (simulate from covariance matrix), 'IRF' (impulse response shock)

		shock_index : int
			Which parameter to apply the shock to if using an IRF.

		shock_value : None or float
			If specified, applies a custom-sized impulse response shock.

		shock_dir : str
			Direction of the IRF shock. One of 'positive' or 'negative'.

		irf_intervals : Boolean
			Whether to have intervals for the IRF plot or not

		Returns
		----------
		A h-length list which contains np.arrays containing shocks for each variable
		"""		
		# Loop over the forecast period

		if shock_type is None:

			random = [np.zeros(self.ylen) for i in range(0,h)]

		elif shock_type == 'IRF':

			cov = self.custom_covariance(self.params)
			post = ss.multivariate_normal(np.zeros(self.ylen),cov)
			if irf_intervals is False:
				random = [np.zeros(self.ylen) for i in range(0,h)]
			else:
				random = [post.rvs() for i in range(0,h)]
				random[0] = np.zeros(self.ylen)

			if shock_value is None:
				if shock_dir=='positive':
					random[0][shock_index] = cov[shock_index,shock_index]**0.5
				elif shock_dir=='negative':
					random[0][shock_index] = -cov[shock_index,shock_index]**0.5
				else:
					raise ValueError("Unknown shock direction!")	
			else:
				random[0][shock_index] = shock_value		

		elif shock_type == 'Cov':
			
			cov = self.custom_covariance(self.params)
			post = ss.multivariate_normal(np.zeros(self.ylen),cov)
			random = [post.rvs() for i in range(0,h)]

		return random

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

	def custom_covariance(self,beta):
		""" Creates Covariance Matrix for a given Beta Vector
		(Not necessarily the OLS covariance)

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		A Covariance Matrix
		"""			

		cov_matrix = np.zeros((self.ylen,self.ylen))

		quick_count = 0
		for i in range(0,self.ylen):
			for k in range(0,self.ylen):
				if i >= k:
					index = self.ylen + self.lags*(self.ylen**2) + quick_count
					quick_count += 1
					cov_matrix[i,k] = self._param_desc[index]['prior'].transform(beta[index])

		return cov_matrix + np.transpose(np.tril(cov_matrix,k=-1))

	def estimator_cov(self,method):
		""" Creates covariance matrix for the estimators

		Parameters
		----------
		method : str
			Estimation method

		Returns
		----------
		A Covariance Matrix
		"""			
		
		Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])	
		Z = self._create_Z(Y)
		if method == 'OLS':
			sigma = self.ols_covariance()
		else:			
			sigma = self.custom_covariance(self.params)
		return np.kron(np.linalg.inv(np.dot(Z,np.transpose(Z))), sigma)

	def irf(self,h=10,shock_index=0,shock_value=None,intervals=True,shock_dir='positive',cumulative=False,**kwargs):

		""" Function allows for mean prediction; also allows shock specification for simulations or impulse response effects

		Parameters
		----------
		h : int
			How many steps ahead to forecast

		shock_index : int
			Which parameter (index) to apply the shock to.

		shock_value : None or float
			If specified, applies a custom-sized impulse response shock.

		intervals : Boolean
			Whether to have intervals for the IRF plot or not

		shock_dir : str
			Direction of the IRF shock. One of 'positive' or 'negative'.

		cumulative : Boolean
			Whether to plot cumulative effect (if no -> plot effect at each step)

		Returns
		----------
		A plot of the impact response
		"""			

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:		

			# Retrieve data, dates and (transformed) parameters
			mu, Y = self._model(self.params)	
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Get steady state values (hacky solution)
			ss_exps = self._forecast_mean(150,t_params,Y,None)
			ss_exps = np.array([i[len(i)-10:len(i)-1] for i in ss_exps])

			# Expectation
			exps = self._forecast_mean(h,t_params,ss_exps,'IRF',shock_index,None,shock_dir)

			if intervals is True:
				# Error bars
				sim_vector = np.array([np.zeros([10000,h]) for i in range(0,self.ylen)])
				for it in range(0,10000):
					exps_sim = self._forecast_mean(h,t_params,Y,'IRF',shock_index,None,shock_dir,True)
					for variable in range(0,self.ylen):
						sim_vector[variable][it,:] = exps_sim[variable][(exps_sim[variable].shape[0]-h):exps_sim[variable].shape[0]]

			for variable in range(exps.shape[0]):

				exp_var = exps[variable][(exps[variable].shape[0]-h-1):exps[variable].shape[0]]
				exp_var = exp_var - ss_exps[variable][ss_exps[variable].shape[0]-1] # demean

				if cumulative is True:
					exp_var = exp_var.cumsum()
					if intervals is True:
						sims = sim_vector[variable]
						sims = [i.cumsum() for i in sims]
						sims = np.transpose(sims)
				elif cumulative is False:
					if intervals is True:
						sims = np.transpose(sim_vector[variable])
						
				plt.figure(figsize=figsize)

				if intervals is True:
					error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(exps[variable],sims,date_index,h,0)

					if cumulative is True:
						forecasted_values = (forecasted_values - ss_exps[variable][ss_exps[variable].shape[0]-1]).cumsum()
					else:
						forecasted_values = (forecasted_values - ss_exps[variable][ss_exps[variable].shape[0]-1])
					
					alpha =[0.15*i/float(100) for i in range(50,12,-2)]
					for count, pre in enumerate(error_bars):

						if cumulative is True:
							ebar = pre - ss_exps[variable][ss_exps[variable].shape[0]-1]					
						else:
							ebar = pre - ss_exps[variable][ss_exps[variable].shape[0]-1]

						plt.fill_between(range(1,len(forecasted_values)), forecasted_values[1:len(ebar)]-ebar[1:len(ebar)], forecasted_values[1:len(ebar)]+ebar[1:len(ebar)],alpha=alpha[count])			
					
				plt.plot(exp_var)
				plt.plot(np.zeros(exp_var.shape[0]),alpha=0.3)	
				if shock_dir == 'positive':				
					plt.title("IR for " + self.data_name[variable] + " for +ve shock in " + self.data_name[shock_index])
				elif shock_dir == 'negative':
					plt.title("IR for " + self.data_name[variable] + " for -ve shock in " + self.data_name[shock_index])					
				plt.xlabel("Time")
				plt.ylabel(self.data_name[variable])
				plt.show()

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

		mu, Y = self._model(beta)
		mu_t = np.transpose(mu)
		Y_t = np.transpose(Y)
		cm = self.custom_covariance(beta)
		diff = Y_t - mu_t

		ll1 =  -(mu_t.shape[0]*mu_t.shape[1]/2.0)*log(2.0*pi) - (mu_t.shape[0]/2.0)*np.linalg.slogdet(cm)[1]
		ll2 = 0

		for t in range(0,mu_t.shape[0]):
			ll2 += np.dot(np.dot(diff[t].T,np.linalg.pinv(cm)),diff[t])

		return -(ll1 -0.5*ll2)

	def ols_covariance(self):
		""" Creates OLS estimate of the covariance matrix

		Returns
		----------
		The OLS estimate of the covariance matrix
		"""			

		Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])		
		return (1.0/(Y[0].shape[0]))*np.dot(self.residuals(Y),np.transpose(self.residuals(Y)))

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
			date_index = self.index[self.lags:self.data[0].shape[0]]
			mu, Y = self._model(self.params)
			for series in range(0,Y.shape[0]):
				plt.figure(figsize=figsize)
				plt.plot(date_index,Y[series],label='Data ' + str(series))
				plt.plot(date_index,mu[series],label='Filter' + str(series),c='black')	
				plt.title(self.data_name[series])
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
			mu, Y = self._model(self.params)	
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Expectation
			exps = self._forecast_mean(h,t_params,Y,None,None)

			# Simulation
			sim_vector = np.array([np.zeros([15000,h]) for i in range(self.ylen)])
			for it in range(0,15000):
				exps_sim = self._forecast_mean(h,t_params,Y,"Cov",None)
				for variable in range(self.ylen):
					sim_vector[variable][it,:] = exps_sim[variable][(exps_sim[variable].shape[0]-h):exps_sim[variable].shape[0]]

			for variable in range(0,exps.shape[0]):
				test = np.transpose(sim_vector[variable])
				error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(exps[variable],test,date_index,h,past_values)

				plt.figure(figsize=figsize)
				if intervals == True:
					alpha = [0.15*i/float(100) for i in range(50,12,-2)]
					for count, pre in enumerate(error_bars):
						plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])			
				plt.plot(plot_index,plot_values)
				plt.title("Forecast for " + self.data_name[variable])
				plt.xlabel("Time")
				plt.ylabel(self.data_name[variable])
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
			new_data = []
			for variable in range(0,self.ylen):
				temp_data = self.data_original.transpose()
				new_data.append(temp_data[variable][0:(temp_data[0].shape[0]-h+t)])
			new_data = np.transpose(np.array(new_data))
			x = VAR(lags=self.lags,integ=self.integ,data=new_data)
			x.fit(printer=False)
			if t == 0:
				predictions = x.predict(1)
			else:
				predictions = pd.concat([predictions,x.predict(1)])
		
		#predictions.rename(columns={0:self.data_name}, inplace=True)
		predictions.index = self.index[(len(self.index)-h):len(self.index)]

		return predictions

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

			# Retrieve data, dates and (transformed) parameters
			mu, Y = self._model(self.params)	
			date_index = self.shift_dates(h)
			t_params = self.transform_parameters()

			# Expectation
			exps = self._forecast_mean(h,t_params,Y,None,None)

			for variable in range(0,exps.shape[0]):
				forecasted_values = exps[variable][(exps[variable].shape[0]-h):exps[variable].shape[0]]
				if variable == 0:
					result = pd.DataFrame(forecasted_values)
					result.rename(columns={0:self.data_name[variable]}, inplace=True)
					result.index = date_index[(len(date_index)-h):len(date_index)]			
				else:
					result[self.data_name[variable]] = forecasted_values	

			return result

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

		date_index = self.index[(self.data[0].shape[0]-h):self.data[0].shape[0]]
		predictions = self.predict_is(h)

		for variable in range(self.ylen):
			plt.figure(figsize=figsize)
			data = self.data[variable][(self.data[0].shape[0]-h):self.data[0].shape[0]]
			plt.plot(date_index,data,label=self.data_name[variable] + ' Data')
			plt.plot(date_index,predictions.ix[:,variable].values,label='Predictions',c='black')
			plt.title(self.data_name[variable])
			plt.legend(loc=2)	
			plt.show()		

	def residuals(self,Y):
		""" Creates the model residuals

		Parameters
		----------
		Y : np.array
			The dependent variables Y

		Returns
		----------
		The model residuals
		"""			

		return (Y-np.dot(self._create_B(Y),self._create_Z(Y)))

