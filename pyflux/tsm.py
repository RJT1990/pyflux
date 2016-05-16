from math import exp, sqrt, log, tanh
import copy
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numdifftools as nd
import pandas as pd

from .covariances import acf
from .inference import BBVI, MetropolisHastings, norm_post_sim, Normal, InverseGamma, Uniform
from .output import TablePrinter
from .tests import find_p_value
from .distributions import q_Normal

class TSM(object):
	""" TSM PARENT CLASS

	Contains general time series methods to be inherited by models.

	Parameters
	----------
	model_type : str
		The type of model (e.g. 'ARIMA', 'GARCH')
	"""

	def __init__(self,model_type):

		# Holding variables for model output
		self.params = []
		self.ses = []
		self.ihessian = []
		self.chains = []
		self.predictions = []

		self._param_desc = [] # Holds information about parameters, such as transforms, priors, etc
		self.model_type = model_type

	def _bbvi_fit(self,posterior,printer=True,step=0.001,iterations=30000):
		""" Performs Black Box Variational Inference

		Parameters
		----------
		posterior : method
			Hands bbvi_fit a posterior object

		printer : Boolean
			Whether to print results or not

		step : float
			Step size for RMSProp

		iterations: int
			How many iterations for BBVI

		Returns
		----------
		None (plots posteriors and stores parameters)
		"""

		# Starting parameters
		phi = self.starting_params.copy()
		self.params = self.starting_params.copy()

		# Starting values for approximate distribution
		for i in range(len(self._param_desc)):
			approx_dist = self._param_desc[i]['q']
			if isinstance(approx_dist, q_Normal):
				self._param_desc[i]['q'].loc = self.params[i]
				if len(self.ses) == 0:
					self._param_desc[i]['q'].scale = -3.0
				else:
					self._param_desc[i]['q'].scale = log(self.ses[i])
		q_list = [k['q'] for k in self._param_desc]
		
		bbvi_obj = BBVI(posterior,q_list,12,step,iterations)
		q, q_params, q_ses = bbvi_obj.lambda_update()
		self.params = q_params

		for k in range(len(self._param_desc)):
			self._param_desc[k]['q'] = q[k]

		self._normal_posterior_sim(self.params,np.diag(np.exp(q_ses)),"Black Box Variational Inference",printer)

	def _laplace_fit(self,obj_type,printer=True):
		""" Performs a Laplace approximation to the posterior

		Parameters
		----------
		obj_type : method
			Whether a likelihood or a posterior

		printer : Boolean
			Whether to print results or not

		Returns
		----------
		None (plots posterior)
		"""

		# Get Mode and Inverse Hessian information
		self.fit(method='MAP',printer=False)

		if len(self.ses) == 0: 
			raise Exception("No Hessian information - Laplace approximation cannot be performed")
		else:
			self._normal_posterior_sim(self.params,self.ihessian,"Laplace",printer)

	def _mcmc_fit(self,scale=(2.38/sqrt(1000)),nsims=100000,printer=True,method="M-H",cov_matrix=None,**kwargs):
		""" Performs MCMC 

		Parameters
		----------
		scale : float
			Default starting scale

		nsims : int
			Number of simulations

		printer : Boolean
			Whether to print results or not

		method : str
			What type of MCMC

		cov_matrix: None or np.array
			Can optionally provide a covariance matrix for M-H.

		Returns
		----------
		None (plots posteriors, stores parameters)
		"""

		figsize = kwargs.get('figsize',(15,15))

		self.fit(method='MAP',printer=False)
		
		if method == "M-H":
			sampler = MetropolisHastings(self.posterior,scale,nsims,self.params,cov_matrix=cov_matrix,model_object=None)
			chain, mean_est, median_est, upper_95_est, lower_95_est = sampler.sample()
		else:
			raise Exception("Method not recognized!")

		self.params = np.asarray(mean_est)

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

		if self.model_type == 'VAR':
			self.data_length = self.data[0]			
		else:
			self.data_length = self.data

		print(self.model_name)
		print("==================")
		print("Method: Metropolis-Hastings")
		print("Number of simulations: " + str(nsims))
		print("Number of observations: " + str(self.data_length.shape[0]-self.max_lag))
		print("Unnormalized Log Posterior: " + str(np.round(-self.posterior(self.params),4)))
		print("")
		print( TablePrinter(fmt, ul='=')(data) )

		fig = plt.figure(figsize=figsize)

		for j in range(len(self.params)):

			for k in range(4):
				iteration = j*4 + k + 1
				ax = fig.add_subplot(len(self.params),4,iteration)

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
					plt.bar(range(1,10),[acf(chain[j],lag) for lag in range(1,10)])
					if iteration == 4:
						plt.title('ACF Plot')						
		sns.plt.show()		

	def _normal_posterior_sim(self,parameters,cov_matrix,method_name,printer):
		""" Simulates from a multivariate normal posterior

		Parameters
		----------
		parameters : np.array
			Contains untransformed starting values for parameters

		cov_matrix : np.array
			The inverse Hessian

		method_name : str
			Name of the estimation procedure using this function

		printer : str
			Whether to print results or not

		Returns
		----------
		None (plots the posteriors)
		"""

		chain, mean_est, median_est, upper_95_est, lower_95_est = norm_post_sim(parameters,cov_matrix)

		for k in range(len(chain)):
			chain[k] = self._param_desc[k]['prior'].transform(chain[k])
			mean_est[k] = self._param_desc[k]['prior'].transform(mean_est[k])
			median_est[k] = self._param_desc[k]['prior'].transform(median_est[k])
			upper_95_est[k] = self._param_desc[k]['prior'].transform(upper_95_est[k])
			lower_95_est[k] = self._param_desc[k]['prior'].transform(lower_95_est[k])				

		mean_est = np.array(mean_est)
		self.chains = chain[:]

		if printer is True:

			data = []

			for i in range(len(self._param_desc)):
				data.append({'param_name': self._param_desc[i]['name'], 'param_mean': np.round(mean_est[i],4), 'param_median': np.round(median_est[i],4), 'ci':  "(" + str(np.round(lower_95_est[i],4)) + " | " + str(np.round(upper_95_est[i],4)) + ")"})

			fmt = [
				('Parameter','param_name',20),
				('Median','param_median',10),
				('Mean', 'param_mean', 15),
				('95% Credibility Interval','ci',25)]

			if self.model_type == 'VAR':
				self.data_length = self.data[0]			
			else:
				self.data_length = self.data

			print(self.model_name)
			print("==================")
			print("Method: " + method_name + " Approximation")
			print("Number of observations: " + str(self.data_length.shape[0]-self.max_lag))				
			print("Unnormalized Log Posterior: " + str(np.round(-self.posterior(mean_est),4)))
			print("")

			print( TablePrinter(fmt, ul='=')(data) )
		
		# Plot densities
		for j in range(len(self.params)):
			fig = plt.figure()
			a = sns.distplot(chain[j], rug=False, hist=False, label=self._param_desc[j]['name'])
			a.set_title(self._param_desc[j]['name'])
			a.set_ylabel('Density')
				
		sns.plt.show()		

	def _ols_fit(self,printer):
		""" Performs OLS

		Parameters
		----------

		printer : Boolean
			Whether to print results or not

		Returns
		----------
		None (stores parameters)
		"""

		self.params = self._create_B_direct().flatten()
		cov = self.ols_covariance()

		# Inelegant - needs refactoring
		for i in range(self.ylen):
			for k in range(self.ylen):
				if i == k:
					self.params = np.append(self.params,self._param_desc[len(self.params)]['prior'].itransform(cov[i,k]))
				elif i > k:
					self.params = np.append(self.params,self._param_desc[len(self.params)]['prior'].itransform(cov[i,k]))

		self.ihessian = self.estimator_cov('OLS')
		self.ses = np.power(np.abs(np.diag(self.ihessian)),0.5)
		t_params = self.params.copy()			
		t_p_std = self.ses.copy() # vector for transformed standard errors
			
		# Create transformed variables
		for k in range(len(t_params)-int(self._param_hide)):
			z_temp = (self.params[k]/float(self.ses[k]))
			t_params[k] = self._param_desc[k]['prior'].transform(t_params[k])
			t_p_std[k] = t_params[k] / z_temp

		# Replace with something more elegant in future versions
		if printer is True:

			data = []

			for i in range(len(self._param_desc)-int(self._param_hide)):
				data.append({'param_name': self._param_desc[i]['name'], 'param_value':np.round(self._param_desc[i]['prior'].transform(self.params[i]),4), 'param_std': np.round(t_p_std[i],4),'param_z': np.round(t_params[i]/float(t_p_std[i]),4),'param_p': np.round(find_p_value(t_params[i]/float(t_p_std[i])),4),'ci': "(" + str(np.round(t_params[i] - t_p_std[i]*1.96,4)) + " | " + str(np.round(t_params[i] + t_p_std[i]*1.96,4)) + ")"})

			fmt = [
			    ('Parameter',       'param_name',   40),
			    ('Estimate',          'param_value',       10),
			    ('Standard Error', 'param_std', 15),
			    ('z',          'param_z',       10),
			    ('P>|z|',          'param_p',       10),
			    ('95% Confidence Interval',          'ci',       25)
				]		

			if self.model_type == 'VAR':
				self.data_length = self.data[0]			
			else:
				self.data_length = self.data

			print(self.model_name)
			print("==================")
			print("Method: OLS")
			print("Number of observations: " + str(self.data_length.shape[0]-self.max_lag))
			print("Log Likelihood: " + str(np.round(-self.likelihood(self.params),4)))
			print("AIC: " + str(np.round(2*len(self.params)+2*self.likelihood(self.params),4)))
			print("BIC: " + str(np.round(2*self.likelihood(self.params) + len(self.params)*log(self.data_length.shape[0]-self.max_lag),4)))
			print("")
			print( TablePrinter(fmt, ul='=')(data) )

	def _optimize_fit(self,obj_type=None,printer=True,**kwargs):
		""" Performs optimization of an objective function

		Parameters
		----------
		obj_type : method
			Whether a likelihood or a posterior

		printer : Boolean
			Whether to print results or not

		Returns
		----------
		None (stores parameters)
		"""

		# Starting parameters
		phi = self.starting_params.copy()
		phi = kwargs.get('start',phi).copy() # If user supplied

		if self.model_type in ['LLT','LLEV']:
			rounding_points=10
		else:
			rounding_points = 4

		# Optimize using L-BFGS-B
		p = optimize.minimize(obj_type,phi,method='L-BFGS-B')
		self.params = p.x.copy()
		t_params = p.x.copy() # vector for transformed parameters (display purposes)			

		# Check that matrix is non-singular; act accordingly
		try:
			self.ihessian = np.linalg.inv(nd.Hessian(obj_type)(self.params))
			self.ses = np.power(np.abs(np.diag(self.ihessian)),0.5)
			t_p_std = self.ses.copy() # vector for transformed standard errors

			# Create transformed variables
			for k in range(len(t_params)-self._param_hide):
				z_temp = (p.x[k]/float(self.ses[k]))
				t_params[k] = self._param_desc[k]['prior'].transform(t_params[k])
				t_p_std[k] = t_params[k] / z_temp

			# Replace with something more elegant in future versions
			if printer is True:

				data = []

				for i in range(len(self._param_desc)-self._param_hide):
					if self._param_desc[i]['prior'].transform == np.array:
						data.append({
							'param_name': self._param_desc[i]['name'], 
							'param_value':np.round(self._param_desc[i]['prior'].transform(p.x[i]),rounding_points), 
							'param_std': np.round(t_p_std[i],rounding_points),
							'param_z': np.round(t_params[i]/float(t_p_std[i]),rounding_points),
							'param_p': np.round(find_p_value(t_params[i]/float(t_p_std[i])),rounding_points),
							'ci': "(" + str(np.round(t_params[i] - t_p_std[i]*1.96,rounding_points)) + " | " + str(np.round(t_params[i] + t_p_std[i]*1.96,rounding_points)) + ")"})
					else:
						data.append({
							'param_name': self._param_desc[i]['name'], 
							'param_value':np.round(self._param_desc[i]['prior'].transform(p.x[i]),rounding_points)})						
				fmt = [
				    ('Parameter',       'param_name',   40),
				    ('Estimate',          'param_value',       10),
				    ('Standard Error', 'param_std', 15),
				    ('z',          'param_z',       10),
				    ('P>|z|',          'param_p',       10),
				    ('95% Confidence Interval',          'ci',       25)
				]

		except: # If Hessian is not invertible...
			if printer is True:
				print ("Hessian not invertible! Consider a different model specification.")
				print ("")

			self.ihessian = []
			self.ses = []

			# Transform parameters
			for k in range(len(t_params)):
				t_params[k] = self._param_desc[k]['prior'].transform(t_params[k])

			if printer is True:

				data = []

				for i in range(len(self._param_desc)):
					data.append({'param_name': self._param_desc[i]['name'], 'param_value':np.round(self._param_desc[i]['prior'].transform(p.x[i]),4)})

				fmt = [
				    ('Parameter',       'param_name',   40),
				    ('Estimate',          'param_value',       10)
				]

		# Final printed output
		if printer is True:

			print(self.model_name)
			print("==================")
			if obj_type == self.likelihood:
				print("Method: MLE")
			elif obj_type == self.posterior:
				print("Method: MAP")

			if self.model_type == 'VAR':
				print("Number of observations: " + str(self.data[0].shape[0]-self.max_lag))
			else:
				print("Number of observations: " + str(self.data.shape[0]-self.max_lag))
			
			print("Log Likelihood: " + str(np.round(-self.likelihood(p.x),4)))
			if obj_type == self.posterior:
				print("Unnormalized Log Posterior: " + str(np.round(-self.posterior(p.x),4)))
			print("AIC: " + str(np.round(2*len(p.x)+2*self.likelihood(p.x),4)))

			if self.model_type == 'VAR':
				print("BIC: " + str(np.round(2*self.likelihood(p.x) + len(p.x)*log(self.data[0].shape[0]-self.max_lag),4)))
			else:
				print("BIC: " + str(np.round(2*self.likelihood(p.x) + len(p.x)*log(self.data.shape[0]-self.max_lag),4)))

			print("")
			print( TablePrinter(fmt, ul='=')(data) )

	def fit(self,method=None,printer=True,**kwargs):
		""" Fits a model

		Parameters
		----------
		method : str
			A fitting method (e.g 'MLE'). Defaults to model specific default method.

		printer : Boolean
			Whether to print results or not

		Returns
		----------
		None (stores fit information)
		"""

		cov_matrix = kwargs.get('cov_matrix',None)
		iterations = kwargs.get('iterations',30000)
		nsims = kwargs.get('nsims',100000)
		step = kwargs.get('step',0.001)

		if method is None:
			method = self.default_method
		elif method not in self.supported_methods:
			raise ValueError("Method not supported!")

		if method == 'MLE':
			self._optimize_fit(self.likelihood,printer,**kwargs)
		elif method == 'MAP':
			self._optimize_fit(self.posterior,printer,**kwargs)	
		elif method == 'M-H':
			self._mcmc_fit(nsims=nsims,method=method,cov_matrix=cov_matrix)
		elif method == "Laplace":
			self._laplace_fit(self.posterior,printer) 
		elif method == "BBVI":
			self._bbvi_fit(self.posterior,printer,step=step,iterations=iterations)
		elif method == "OLS":
			self._ols_fit(printer)			

	def posterior(self,beta):
		""" Returns negative log posterior

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		Negative log posterior
		"""

		post = self.likelihood(beta)
		for k in range(0,self.param_no):
			post += -self._param_desc[k]['prior'].logpdf(beta[k])
		return post

	def adjust_prior(self,index,prior):
		""" Adjusts priors for the parameters

		Parameters
		----------
		index : int or list[int]
			Which parameter index/indices to be altered

		prior : PRIOR object
			Which prior distribution? E.g. Normal(0,1)

		Returns
		----------
		None (changes priors in self._param_desc)
		"""

		if isinstance(index, list):
			for item in index:
				if item < 0 or item > (self.param_no-1) or not isinstance(item, int):
					raise ValueError("Oops - the parameter index " + str(item) + " you have entered is invalid!")
				else:
					self._param_desc[item]['prior'] = prior				
		else:
			if index < 0 or index > (self.param_no-1) or not isinstance(index, int):
				raise ValueError("Oops - the parameter index " + str(index) + " you have entered is invalid!")
			else:
				self._param_desc[index]['prior'] = prior


	def list_priors(self):
		""" Lists the current prior specification

		Returns
		----------
		None (prints out prior information)
		"""

		prior_list = []
		prior_desc = []
		param_trans = []

		for i in range(self.param_no):
			param_trans.append(self._param_desc[i]['prior'].transform_name)
			x = self._param_desc[i]['prior']
			if isinstance(x, Normal):
				prior_list.append('Normal')
				prior_desc.append('mu0: ' + str(self._param_desc[i]['prior'].mu0) + ', sigma0: ' + str(self._param_desc[i]['prior'].sigma0))
			elif isinstance(x, InverseGamma):
				prior_list.append('Inverse Gamma')
				prior_desc.append('alpha: ' + str(self._param_desc[i]['prior'].alpha) + ', beta: ' + str(self._param_desc[i]['prior'].beta))
			elif isinstance(x, Uniform):
				prior_list.append('Uniform')
				prior_desc.append('n/a (non-informative)')
			else:
				raise ValueError("Error - prior distribution not detected!")

		data = []

		for i in range(self.param_no):
			data.append({'param_index': len(data), 'param_name': self._param_desc[i]['name'], 'param_prior': prior_list[i], 'param_prior_params': prior_desc[i], 'param_trans': param_trans[i]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',25),
			('Prior Type','param_prior',10),
			('Prior Parameters', 'param_prior_params', 25),
			('Transformation','param_trans',20)]
		
		print( TablePrinter(fmt, ul='=')(data) )

	def list_q(self):
		""" Lists the current approximating distributions for variational inference

		Returns
		----------
		None (lists prior specification)
		"""

		q_list = []

		for i in range(self.param_no):
			x = self._param_desc[i]['q']
			if isinstance(x, q_Normal):
				q_list.append('Normal')
			elif isinstance(x, q_InverseGamma):
				q_list.append('Inverse Gamma')
			elif isinstance(x, q_Uniform):
				q_list.append('Uniform')
			else:
				raise Exception("Error - prior distribution not detected!")

		data = []

		for i in range(self.param_no):
			data.append({'param_index': len(data), 'param_name': self._param_desc[i]['name'], 'param_q': q_list[i]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',25),
			('q(z)','param_q',10)]

		print("Approximate distributions for mean-field variational inference")
		print("")
		print( TablePrinter(fmt, ul='=')(data) )

	def shift_dates(self,h):
		""" Auxiliary function for creating dates for forecasts

		Parameters
		----------
		h : int
			How many steps to forecast

		Returns
		----------
		A transformed date_index object
		"""		

		date_index = copy.deepcopy(self.index)
		date_index = date_index[self.max_lag:len(date_index)]

		if self.is_pandas is True:

			if isinstance(date_index,pd.tseries.index.DatetimeIndex):

				# Only configured for days - need to support smaller time intervals!
				for t in range(h):
					date_index += pd.DateOffset((date_index[len(date_index)-1] - date_index[len(date_index)-2]).days)

			elif isinstance(date_index,pd.core.index.Int64Index):

				for i in range(h):
					new_value = date_index.values[len(date_index.values)-1] + (date_index.values[len(date_index.values)-1] - date_index.values[len(date_index.values)-2])
					date_index = pd.Int64Index(np.append(date_index.values,new_value))

		else:

			for t in range(h):
				date_index.append(date_index[len(date_index)-1]+1)

		return date_index	

	def transform_parameters(self):
		""" Transforms parameters to actual scale by applying link function

		Returns
		----------
		Transformed parameters 
		"""		
		trans_params = copy.deepcopy(self.params)
		for k in range(len(self.params)):
			trans_params[k] = self._param_desc[k]['prior'].transform(self.params[k])	
		return trans_params	