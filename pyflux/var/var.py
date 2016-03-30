from .. import inference as ifr
from .. import distributions as dist
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import exp, sqrt, log, tanh, pi
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from .. import covariances as cov
import numdifftools as nd
import datetime

class VAR(tsm.TSM):

	def __init__(self,data,lags,target=None,integ=0):

		# Initialize TSM object
		tsm.TSM.__init__(self,'VAR')

		# Parameters
		self.lags = lags
		self.int = integ

		# Check Data format
		if isinstance(data, pd.DataFrame):
			self.index = data.index		
			self.data = data.values
			self.data_name = data.columns.values
			self.data_type = 'pandas'
			print str(self.data_name) + " picked as target variables"
			print ""

		elif isinstance(data, np.ndarray):
			self.data_name = np.asarray(range(1,len(data)+1))	
			self.data_type = 'numpy'	
			self.data = data
			self.index = range(len(data[0]))

		else:
			raise Exception("The data input is not pandas or numpy compatible!")

		# Difference data

		X = np.transpose(self.data)
		for order in range(self.int):
			X = np.asarray([np.diff(i) for i in X])
			self.data_name = np.asarray(["Differenced " + str(i) for i in self.data_name])
		self.data = X		

		self.ylen = len(self.data_name)

		# Create VAR parameters
		for variable in range(self.ylen):
			self.param_desc.append({'name' : self.data_name[variable] + ' Constant', 'index': len(self.param_desc), 'prior': ifr.Normal(0,3,transform=None), 'q': dist.Normal(0,3)})		
			other_variables = np.delete(range(self.ylen), [variable])
			for lag_no in range(self.lags):
				self.param_desc.append({'name' : str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', 'index': len(self.param_desc), 'prior': ifr.Normal(0,0.5,transform=None), 'q': dist.Normal(0,3)})
				for other in other_variables:
					self.param_desc.append({'name' : str(self.data_name[other]) + ' to ' + str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', 'index': len(self.param_desc), 'prior': ifr.Normal(0,0.5,transform=None), 'q': dist.Normal(0,3)})

		# Variance prior
		for i in range(self.ylen):
			for k in range(self.ylen):
				if i == k:
					self.param_desc.append({'name' : 'Sigma' + str(self.data_name[i]),'index': len(self.param_desc),'prior': ifr.Uniform(transform='exp'), 'q': dist.Normal(0,3)})
				elif i > k:
					self.param_desc.append({'name' : 'Sigma' + str(self.data_name[i]) + ' to ' + str(self.data_name[k]),'index': len(self.param_desc), 'prior': ifr.Uniform(transform=None), 'q': dist.Normal(0,3)})

		# Other attributes

		self.param_hide = len(self.data)**2 - (len(self.data)**2 - len(self.data))/2 # Whether to cutoff variance parameters from results
		self.param_no = len(self.param_desc)
		self.supported_methods = ["OLS","MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "OLS"

	# Holds the core model matrices
	def model(self,beta,x):
		Y = np.array([reg[self.lags:len(reg)] for reg in x])
		# Transform parameters
		beta = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		params = []
		col_length = 1 + self.ylen*self.lags
		for i in range(self.ylen):
			params.append(beta[(col_length*i): (col_length*(i+1))])

		mu = np.dot(np.array(params),self.create_Z())

		return mu, Y

	# Negative loglikelihood for model (need to vectorize for loop)
	def likelihood(self,beta):
		mu, Y = self.model(beta,self.data)
		mu_t = np.transpose(mu)
		Y_t = np.transpose(Y)
		loglik = np.array([-ss.multivariate_normal.logpdf(mu_t[t], Y_t[t], self.create_cov(beta)) for t in range(len(Y[0]))])
		return np.sum(loglik)

	# Creates design matrix
	def create_Z(self):
		Y = np.array([reg[self.lags:len(reg)] for reg in self.data])
		Z = np.ones(((self.ylen*self.lags +1),len(Y[0])))
		row_count = 1
		for lag in range(1,self.lags+1):
			for reg in range(len(Y)):
				Z[row_count,:] = self.data[reg][(self.lags-lag):len(self.data[reg])-lag]			
				row_count += 1
		return Z

	# Creates coefficient matrix
	def create_B(self):
		Y = np.array([reg[self.lags:len(reg)] for reg in self.data])
		Z = self.create_Z()
		return np.dot(np.dot(Y,np.transpose(Z)),np.linalg.inv(np.dot(Z,np.transpose(Z))))

	# Create model residuals
	def create_eps(self):
		Y = np.array([reg[self.lags:len(reg)] for reg in self.data])
		return (Y-np.dot(self.create_B(),self.create_Z()))

	# OLS estimate of covariance matrix
	def estimate_cov(self):
		Y = np.array([reg[self.lags:len(reg)] for reg in self.data])		
		return (1.0/(len(Y[0])))*np.dot(self.create_eps(),np.transpose(self.create_eps()))

	# Creates covariance matrix given a beta vector
	def create_cov(self,beta):

		cov_matrix = np.zeros((self.ylen,self.ylen))

		quick_count = 0
		for i in range(self.ylen):
			for k in range(self.ylen):
				if i >= k:
					index = self.ylen + self.lags*(self.ylen**2) + quick_count
					quick_count += 1
					cov_matrix[i,k] = self.param_desc[index]['prior'].transform(beta[index])

		return cov_matrix + np.transpose(np.tril(cov_matrix,k=-1))

	# Creates estimator covariance matrix
	def estimator_cov(self,method):
		Z = self.create_Z()
		if method ==  'MLE':
			sigma = self.create_cov(self.params)
		else:
			sigma = self.estimate_cov()
		return np.kron(np.linalg.inv(np.dot(Z,np.transpose(Z))), sigma)

	def forecast_mean(self,T,params_to_use,exp_values,shock_type=None,shock_index=0,shock_value=None,irf_interval=None,shock_dir='positive'):

		# Random shocks
		random = []
		cov = self.create_cov(self.params)
		post = ss.multivariate_normal(np.zeros(self.ylen),cov)
		
		for t in range(T):
			if shock_type is None:
				random.append(np.zeros(self.ylen))
			elif shock_type == "Cov":
				random.append(post.rvs())
			elif shock_type == 'IRF':
				if t == 0:
					irf_values = np.zeros(self.ylen)
					if shock_value is None:
						if shock_dir=='positive':
							irf_values[shock_index] = cov[shock_index,shock_index]**0.5
						elif shock_dir=='negative':
							irf_values[shock_index] = -cov[shock_index,shock_index]**0.5
						else:
							raise ValueError("Unknown shock direction!")							
					else:
						irf_values[shock_index] = shock_value
					random.append(irf_values)
				else:
					if irf_interval is None:
						random.append(np.zeros(self.ylen))
					else:
						random.append(post.rvs())

		# Create list of variables
		exp = []
		for variable in range(self.ylen):
			exp.append(exp_values[variable])
		
		# Each forward projection
		for t in range(T):
			new_values = np.zeros(self.ylen)

			# Each variable
			for variable in range(self.ylen):
				index_ref = variable*(1+self.ylen*self.lags)
				new_value = params_to_use[index_ref] # constant

				# VAR(p) terms
				for lag in range(self.lags):
					for lagged_var in range(self.ylen):
						new_value += params_to_use[index_ref+lagged_var+(lag*self.ylen)+1]*exp[lagged_var][len(exp[lagged_var])-1-lag]
				
				# Random shock
				new_value += random[t][variable]
				new_values[variable] = new_value

			# Add new values
			for variable in range(self.ylen):
				exp[variable] = np.append(exp[variable],new_values[variable])

		return np.array(exp)


	def irf(self,T=10,shock_index=0,shock_value=None,intervals=True,shock_dir='positive'):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:		

			# Get data
			exp_values = np.array([reg[self.lags:len(reg)] for reg in self.data])
			mu, Y = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Get steady state values (hacky)
			ss_exps = self.forecast_mean(150,params_to_use,exp_values,None)
			ss_exps = np.array([i[len(i)-10:len(i)-1] for i in ss_exps])

			# Expectation
			exps = self.forecast_mean(T,params_to_use,ss_exps,'IRF',shock_index,None,None,shock_dir)

			if intervals is True:
				# Error bars
				sim_vector = np.array([np.zeros([10000,T]) for i in range(self.ylen)])
				for it in range(10000):
					exps_sim = self.forecast_mean(T,params_to_use,exp_values,'IRF',shock_index,None,'On',shock_dir)
					for variable in range(self.ylen):
						sim_vector[variable][it,:] = exps_sim[variable][(len(exps_sim[variable])-T):(len(exps_sim[variable]))]

			for variable in range(len(exps)):
				
				if intervals is True:
					test = np.transpose(sim_vector[variable])
					error_bars = [np.percentile(i,95) for i in test] - exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
					error_bars_90 = [np.percentile(i,90) for i in test] - exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
					forecasted_values = exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
					
				exp_values = exps[variable][(len(exps[variable])-T-1):len(exps[variable])]

				# Plot prediction graph		
				if intervals is True:
					plt.fill_between(range(1,len(forecasted_values)+1), forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
					plt.fill_between(range(1,len(forecasted_values)+1), forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)						
				plt.plot(exp_values)
				plt.plot(ss_exps[variable][len(ss_exps[variable])-1]*np.ones(len(exp_values)),alpha=0.3)					
				plt.title("IR for " + self.data_name[variable] + " for +ve shock in " + self.data_name[shock_index])
				plt.xlabel("Time")
				plt.ylabel(self.data_name[variable])
				plt.show()

	# Produces T-step ahead forecast for the series
	# This code is very inefficient; needs amending
	def predict(self,T=5,lookback=20,intervals=True):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Get data
			exp_values = np.array([reg[self.lags:len(reg)] for reg in self.data])
			date_index = self.index[self.lags:len(self.data[0])]

			for t in range(T):
				if self.data_type == 'pandas':
					date_index += pd.DateOffset(1)
				elif self.data_type == 'numpy':
					date_index.append(date_index[len(date_index)-1]+1)


			mu, Y = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Expectation
			exps = self.forecast_mean(T,params_to_use,exp_values,None,None)

			# Error bars
			sim_vector = np.array([np.zeros([10000,T]) for i in range(self.ylen)])
			for it in range(10000):
				exps_sim = self.forecast_mean(T,params_to_use,exp_values,"Cov",None)
				for variable in range(self.ylen):
					sim_vector[variable][it,:] = exps_sim[variable][(len(exps_sim[variable])-T):(len(exps_sim[variable]))]

			for variable in range(len(exps)):
				test = np.transpose(sim_vector[variable])
				error_bars = [np.percentile(i,95) for i in test] - exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
				error_bars_90 = [np.percentile(i,90) for i in test] - exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
				forecasted_values = exps[variable][(len(exps[variable])-T):(len(exps[variable]))]
				exp_values = exps[variable][len(exps[variable])-T-lookback:len(exps[variable])]
				date_index = date_index[len(date_index)-T-lookback:len(date_index)]

				# Plot prediction graph	
				if intervals is True:
					plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
					plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)						
				plt.plot(date_index,exp_values)
				plt.title("Forecast for " + self.data_name[variable])
				plt.xlabel("Time")
				plt.ylabel(self.data_name[variable])
				plt.show()

