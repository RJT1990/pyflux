import inference as ifr
import arma as arma
import output as op
import tests as tst
import distributions as dst
import numpy as np
from math import exp, sqrt, log, tanh
from scipy import optimize
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numdifftools as nd
import covariances as cov
import pandas as pd

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

		# Holds parameter descriptions and model type
		self.param_desc = []
		self.model_type = model_type

	def posterior(self,beta):
		""" Returns negative log posterior

		Parameters
		----------
		beta : np.ndarray
			Contains untransformed starting values for parameters

		Returns
		----------
		Negative log posterior
		"""

		post = self.likelihood(beta)
		for k in range(self.param_no):
			post += -self.param_desc[k]['prior'].logpdf(beta[k])
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
		None (changes priors in self.param_desc)
		"""

		if isinstance(index, list):
			for item in index:
				if item < 0 or item > (self.param_no-1) or not isinstance(item, int):
					raise Exception("Oops - the parameter index " + str(item) + " you have entered is invalid!")
				else:
					self.param_desc[item]['prior'] = prior				
		else:
			if index < 0 or index > (self.param_no-1) or not isinstance(index, int):
				raise Exception("Oops - the parameter index " + str(index) + " you have entered is invalid!")
			else:
				self.param_desc[index]['prior'] = prior

	# Wrapper function for inference options
	def fit(self,method=None,printer=True,nsims=100000,cov_matrix=None,step=0.001,iterations=30000,**kwargs):
		""" Fits a model

		Parameters
		----------
		method : str
			A fitting method (e.g 'MLE')

		printer : Boolean
			Whether to print results or not

		nsims : int (optional)
			How many simulations if an MCMC method

		cov_matrix : None or np.ndarray
			Option to include covariance matrix for M-H MCMC

		step : float
			Step size for BBVI method

		iterations = int
			How many iterations for BBVI

		Returns
		----------
		None (stores fit information)
		"""

		if method is None:
			method = self.default_method

		if method not in self.supported_methods:
			raise ValueError("Method not supported!")

		if method == 'MLE':
			self.optimize_fit(self.likelihood,printer)
		elif method == 'MAP':
			self.optimize_fit(self.posterior,printer)	
		elif method == 'M-H':
			self.mcmc_fit(nsims=nsims,method=method,cov_matrix=cov_matrix)
		elif method == "Laplace":
			self.laplace_fit(self.posterior,printer) 
		elif method == "BBVI":
			self.bbvi_fit(self.posterior,printer,step=step,iterations=iterations)
		elif method == "OLS":
			self.ols_fit(printer)			
		else:
			raise ValueError("Method not recognized!")

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
			param_trans.append(self.param_desc[i]['prior'].transform_name)
			x = self.param_desc[i]['prior']
			if isinstance(x, ifr.Normal):
				prior_list.append('Normal')
				prior_desc.append('mu0: ' + str(self.param_desc[i]['prior'].mu0) + ', sigma0: ' + str(self.param_desc[i]['prior'].sigma0))
			elif isinstance(x, ifr.InverseGamma):
				prior_list.append('Inverse Gamma')
				prior_desc.append('alpha: ' + str(self.param_desc[i]['prior'].alpha) + ', beta: ' + str(self.param_desc[i]['prior'].beta))
			elif isinstance(x, ifr.Uniform):
				prior_list.append('Uniform')
				prior_desc.append('n/a (non-informative)')
			else:
				raise Exception("Error - prior distribution not detected!")

		data = []

		for i in range(self.param_no):
			data.append({'param_index': len(data), 'param_name': self.param_desc[i]['name'], 'param_prior': prior_list[i], 'param_prior_params': prior_desc[i], 'param_trans': param_trans[i]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',25),
			('Prior Type','param_prior',10),
			('Prior Parameters', 'param_prior_params', 25),
			('Transformation','param_trans',20)]
		
		print( op.TablePrinter(fmt, ul='=')(data) )

	def list_q(self):
		""" Lists the current approximating distributions for variational inference

		Returns
		----------
		None (lists prior specification)
		"""

		q_list = []

		for i in range(self.param_no):
			x = self.param_desc[i]['q']
			if isinstance(x, dst.Normal):
				q_list.append('Normal')
			elif isinstance(x, dst.InverseGamma):
				q_list.append('Inverse Gamma')
			elif isinstance(x, dst.Uniform):
				q_list.append('Uniform')
			else:
				raise Exception("Error - prior distribution not detected!")

		data = []

		for i in range(self.param_no):
			data.append({'param_index': len(data), 'param_name': self.param_desc[i]['name'], 'param_q': q_list[i]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',25),
			('q(z)','param_q',10)]
		print "Approximate distributions for mean-field variational inference"
		print ""
		print( op.TablePrinter(fmt, ul='=')(data) )

	def normal_posterior_sim(self,parameters,cov_matrix,method_name,printer):
		""" Simulates from a multivariate normal posterior

		Parameters
		----------
		parameters : np.ndarray
			Contains untransformed starting values for parameters

		cov_matrix : np.ndarray
			The inverse Hessian

		method_name : str
			Name of the estimation procedure using this function

		printer : str
			Whether to print results or not

		Returns
		----------
		None (plots the posteriors)
		"""

		chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.norm_post_sim(parameters,cov_matrix)

		# Transform parameters - replace if statement with transform information directly in future version
		for k in range(len(chain)):
			if self.param_desc[k]['prior'].transform == np.exp:
				chain[k] = np.exp(chain[k])
				mean_est[k] = exp(mean_est[k])
				median_est[k] = exp(median_est[k])
				upper_95_est[k] = exp(upper_95_est[k])
				lower_95_est[k] = exp(lower_95_est[k])				
			elif self.param_desc[k]['prior'].transform == np.tanh: 
				chain[k] = np.tanh(chain[k])
				mean_est[k] = tanh(mean_est[k])
				median_est[k] = tanh(median_est[k])
				upper_95_est[k] = tanh(upper_95_est[k])
				lower_95_est[k] = tanh(lower_95_est[k])	

		self.chains = chain

		if printer is True:

			data = []

			for i in range(len(self.param_desc)):
				data.append({'param_name': self.param_desc[i]['name'], 'param_mean': round(mean_est[i],4), 'param_median': round(median_est[i],4), 'ci':  "(" + str(round(lower_95_est[i],4)) + " | " + str(round(upper_95_est[i],4)) + ")"})

			fmt = [
				('Parameter','param_name',20),
				('Median','param_median',10),
				('Mean', 'param_mean', 15),
				('95% Credibility Interval','ci',25)]

			if self.model_type == 'ARIMA':
				self.model_name = "ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ") regression"
				self.cutoff = max(self.ar,self.ma)
			elif self.model_type == 'GAS':
				self.model_name = "GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") regression"
				self.cutoff = max(self.ar,self.sc)
			elif self.model_type == 'VAR':
				self.model_name = "VAR(" + str(self.lags) + ") regression"
				self.cutoff = self.lags			
			elif self.model_type in ['EGARCH', 'GARCH']:
				self.model_name = self.model_type + "(" + str(self.p) + "," + str(self.q) + ") regression"
				self.cutoff = max(self.p,self.q)	
			elif self.model_type == 'GPNARX':
				self.model_name = "GP-NARX(" + str(self.ar) + ") regression"
				self.cutoff = self.max_lag	

			print self.model_name 
			print "=================="
			print "Method: " + method_name + " Approximation"
			print "Number of observations: " + str(len(self.data)-self.cutoff)				
			print "Unnormalized Log Posterior: " + str(round(-self.posterior(mean_est),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )
		
		# Plot densities
		for j in range(len(self.params)):
			fig = plt.figure()
			a = sns.distplot(chain[j], rug=False, hist=False, label=self.param_desc[j]['name'])
			a.set_title(self.param_desc[j]['name'])
			a.set_ylabel('Density')
				
		sns.plt.show()		

	def laplace_fit(self,obj_type,printer=True):
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

		if len(self.ses) == 0: # no errors, no Laplace party
			raise Exception("No Hessian information - Laplace approximation cannot be performed")
		else:
			self.normal_posterior_sim(self.params,self.ihessian,"Laplace",printer)

	def optimize_fit(self,obj_type,printer=True):
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
		phi = np.zeros(self.param_no)
		rounding_points = 4

		# Starting values
		if self.model_type == 'GAS':
			if self.dist in ['Laplace','Normal']:
				phi[0] = np.mean(self.data)
			elif self.dist == 'Poisson':
				phi[0] = np.log(np.mean(self.data))
		elif self.model_type == 'VAR':
			phi = self.create_B_direct().flatten()
			cov = self.estimate_cov()

			# Inelegant - needs refactoring
			for i in range(self.ylen):
				for k in range(self.ylen):
					if i == k:
						phi = np.append(phi,self.param_desc[len(phi)]['prior'].itransform(cov[i,k]))
					elif i > k:
						phi = np.append(phi,self.param_desc[len(phi)]['prior'].itransform(cov[i,k]))			
		elif self.model_type == 'GARCH':
			phi = np.ones(self.param_no)*0.00001
			phi[0] = self.param_desc[0]['prior'].itransform(np.mean(np.power(self.data,2)))
		elif self.model_type == 'EGARCH':
			phi[0] = self.param_desc[0]['prior'].itransform(np.log(np.mean(np.power(self.data,2))))
		elif self.model_type == 'GPNARX':
			phi = np.ones(self.param_no)*-1.0
			arma_start = arma.ARIMA(self.data,ar=self.ar,ma=0,integ=self.integ)
			arma_start.fit(printer=False)
			phi[0] = log(exp(arma_start.params[len(arma_start.params)-1])**2)			
		elif self.model_type in ['LLT','LLEV']:
			phi = np.ones(self.param_no)*-0.0
			rounding_points=10

		# Optimize using L-BFGS-B
		p = optimize.minimize(obj_type,phi,method='L-BFGS-B')
		self.params = copy.deepcopy(p.x)

		# Vector for transformed parameters
		t_params = copy.deepcopy(p.x)			

		# Check that matrix is non-singular; act accordingly
		try:
			self.ihessian = np.linalg.inv(nd.Hessian(obj_type)(self.params))
			self.ses = np.abs(np.diag(self.ihessian))**0.5

			t_p_std = copy.deepcopy(self.ses) # vector for transformed standard errors
			
			# Create transformed variables
			for k in range(len(t_params)-self.param_hide):
				z_temp = (p.x[k]/float(self.ses[k]))
				t_params[k] = self.param_desc[k]['prior'].transform(t_params[k])
				t_p_std[k] = t_params[k] / z_temp

			# Replace with something more elegant in future versions
			if printer is True:

				data = []

				for i in range(len(self.param_desc)-self.param_hide):
					if self.param_desc[i]['prior'].transform == np.array:
						data.append({
							'param_name': self.param_desc[i]['name'], 
							'param_value':round(self.param_desc[i]['prior'].transform(p.x[i]),rounding_points), 
							'param_std': round(t_p_std[i],rounding_points),
							'param_z': round(t_params[i]/float(t_p_std[i]),rounding_points),
							'param_p': round(tst.find_p_value(t_params[i]/float(t_p_std[i])),rounding_points),
							'ci': "(" + str(round(t_params[i] - t_p_std[i]*1.96,rounding_points)) + " | " + str(round(t_params[i] + t_p_std[i]*1.96,rounding_points)) + ")"})
					else:
						data.append({
							'param_name': self.param_desc[i]['name'], 
							'param_value':round(self.param_desc[i]['prior'].transform(p.x[i]),rounding_points)})						
				fmt = [
				    ('Parameter',       'param_name',   40),
				    ('Estimate',          'param_value',       10),
				    ('Standard Error', 'param_std', 15),
				    ('z',          'param_z',       10),
				    ('P>|z|',          'param_p',       10),
				    ('95% Confidence Interval',          'ci',       25)
				]

		except: # If Hessian is not invertible...
			print "Hessian not invertible! Consider a different model specification."
			print ""

			# Reset iHessian and standard errors, reflecting non-invertibility
			self.ihessian = []
			self.ses = []

			# Transform parameters
			for k in range(len(t_params)):
				t_params[k] = self.param_desc[k]['prior'].transform(t_params[k])

			if printer is True:

				data = []

				for i in range(len(self.param_desc)):
					data.append({'param_name': self.param_desc[i]['name'], 'param_value':round(self.param_desc[i]['prior'].transform(p.x[i]),4)})

				fmt = [
				    ('Parameter',       'param_name',   40),
				    ('Estimate',          'param_value',       10)
				]

		# Final printed output
		if printer is True:

			if self.model_type == 'ARIMA':
				self.model_name = "ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ") regression"
				self.cutoff = self.max_lag	
				self.data_length = self.data
			elif self.model_type == 'GAS':
				self.model_name = "GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") regression"
				self.cutoff = self.max_lag	
				self.data_length = self.data				
			elif self.model_type == 'VAR':
				self.model_name = "VAR(" + str(self.lags) + ") regression"
				self.cutoff = self.lags			
				self.data_length = self.data[0]
			elif self.model_type in ['EGARCH', 'GARCH']:
				self.model_name = self.model_type + "(" + str(self.p) + "," + str(self.q) + ") regression"
				self.cutoff = self.max_lag		
				self.data_length = self.data	
			elif self.model_type == 'GPNARX':
				self.model_name = "GP-NARX(" + str(self.max_lag) + ") regression"
				self.cutoff = self.max_lag			
				self.data_length = self.data						

			print self.model_name 
			print "=================="
			if obj_type == self.likelihood:
				print "Method: MLE"
			elif obj_type == self.posterior:
				print "Method: MAP"
			print "Number of observations: " + str(len(self.data_length)-self.cutoff)
			print "Log Likelihood: " + str(round(-self.likelihood(p.x),4))
			if obj_type == self.posterior:
				print "Unnormalized Log Posterior: " + str(round(-self.posterior(p.x),4))
			print "AIC: " + str(round(2*len(p.x)+2*self.likelihood(p.x),4))
			print "BIC: " + str(round(2*self.likelihood(p.x) + len(p.x)*log(len(self.data_length)-self.cutoff),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )

	def ols_fit(self,printer):
		""" Performs OLS

		Parameters
		----------

		printer : Boolean
			Whether to print results or not

		Returns
		----------
		None (stores parameters)
		"""

		self.params = self.create_B_direct().flatten()
		cov = self.estimate_cov()

		# Inelegant - needs refactoring
		for i in range(self.ylen):
			for k in range(self.ylen):
				if i == k:
					self.params = np.append(self.params,self.param_desc[len(self.params)]['prior'].itransform(cov[i,k]))
				elif i > k:
					self.params = np.append(self.params,self.param_desc[len(self.params)]['prior'].itransform(cov[i,k]))

		self.ihessian = self.estimator_cov('OLS')
		self.ses = np.abs(np.diag(self.ihessian))**0.5
		t_params = copy.deepcopy(self.params)			
		t_p_std = copy.deepcopy(self.ses) # vector for transformed standard errors
			
		# Create transformed variables
		for k in range(len(t_params)-self.param_hide):
			z_temp = (self.params[k]/float(self.ses[k]))
			t_params[k] = self.param_desc[k]['prior'].transform(t_params[k])
			t_p_std[k] = t_params[k] / z_temp

		# Replace with something more elegant in future versions
		if printer is True:

			data = []

			for i in range(len(self.param_desc)-self.param_hide):
				data.append({'param_name': self.param_desc[i]['name'], 'param_value':round(self.param_desc[i]['prior'].transform(self.params[i]),4), 'param_std': round(t_p_std[i],4),'param_z': round(t_params[i]/float(t_p_std[i]),4),'param_p': round(tst.find_p_value(t_params[i]/float(t_p_std[i])),4),'ci': "(" + str(round(t_params[i] - t_p_std[i]*1.96,4)) + " | " + str(round(t_params[i] + t_p_std[i]*1.96,4)) + ")"})

			fmt = [
			    ('Parameter',       'param_name',   40),
			    ('Estimate',          'param_value',       10),
			    ('Standard Error', 'param_std', 15),
			    ('z',          'param_z',       10),
			    ('P>|z|',          'param_p',       10),
			    ('95% Confidence Interval',          'ci',       25)
				]		

			if self.model_type == 'ARIMA':
				self.model_name = "ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ") regression"
				self.cutoff = max(self.ar,self.ma)
				self.data_length = self.data			
			elif self.model_type == 'VAR':
				self.model_name = "VAR(" + str(self.lags) + ") regression"
				self.cutoff = self.lags			
				self.data_length = self.data[0]

			print self.model_name 
			print "=================="
			print "Method: OLS"
			print "Number of observations: " + str(len(self.data_length)-self.cutoff)
			print "Log Likelihood: " + str(round(-self.likelihood(self.params),4))
			print "AIC: " + str(round(2*len(self.params)+2*self.likelihood(self.params),4))
			print "BIC: " + str(round(2*self.likelihood(self.params) + len(self.params)*log(len(self.data_length)-self.cutoff),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )

	def mcmc_fit(self,scale=(2.38/sqrt(10000)),nsims=100000,printer=True,method="M-H",cov_matrix=None,**kwargs):
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

		cov_matrix: None or np.ndarray
			Can optionally provide a covariance matrix for M-H.

		Returns
		----------
		None (plots posteriors, stores parameters)
		"""

		figsize = kwargs.get('figsize',(15,15))

		self.fit(method='MAP',printer=False)
		
		if method == "M-H":
			chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.metropolis_hastings(self.posterior,scale,nsims,self.params,cov_matrix=cov_matrix)
		else:
			raise Exception("Method not recognized!")

		self.params = np.asarray(mean_est)

		for k in range(len(chain)):
			if self.param_desc[k]['prior'].transform == np.exp:
				chain[k] = np.exp(chain[k])
				mean_est[k] = exp(mean_est[k])
				median_est[k] = exp(median_est[k])
				upper_95_est[k] = exp(upper_95_est[k])
				lower_95_est[k] = exp(lower_95_est[k])				
			elif self.param_desc[k]['prior'].transform == np.tanh: 
				chain[k] = np.tanh(chain[k])
				mean_est[k] = tanh(mean_est[k])
				median_est[k] = tanh(median_est[k])				
				upper_95_est[k] = tanh(upper_95_est[k])
				lower_95_est[k] = tanh(lower_95_est[k])	

		self.chains = chain

		if printer == True:

			data = []

			for i in range(len(self.param_desc)):
				data.append({'param_name': self.param_desc[i]['name'], 'param_mean':round(mean_est[i],4), 'param_median':round(median_est[i],4), 'ci': "(" + str(round(lower_95_est[i],4)) + " | " + str(round(upper_95_est[i],4)) + ")"})

		fmt = [
			('Parameter','param_name',20),
			('Median','param_median',10),
			('Mean', 'param_mean', 15),
			('95% Credibility Interval','ci',25)]

		if self.model_type == 'ARIMA':
			self.model_name = "ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ") regression"
			self.cutoff = max(self.ar,self.ma)
		elif self.model_type == 'GAS':
			self.model_name = "GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") regression"
			self.cutoff = max(self.ar,self.sc)
		elif self.model_type == 'VAR':
			self.model_name = "VAR(" + str(self.lags) + ") regression"
			self.cutoff = self.lags						
		elif self.model_type in ['EGARCH', 'GARCH']:
			self.model_name = self.model_type + "(" + str(self.p) + "," + str(self.q) + ") regression"
			self.cutoff = max(self.p,self.q)
		elif self.model_type == 'GPNARX':
			self.model_name = "GP-NARX(" + str(self.ar) + ") regression"
			self.cutoff = self.max_lag	

		print self.model_name 
		print "=================="
		print "Method: Metropolis-Hastings"
		print "Number of simulations: " + str(nsims)
		print "Number of observations: " + str(len(self.data)-self.cutoff)
		print "Unnormalized Log Posterior: " + str(round(-self.posterior(mean_est),4))
		print ""
		print( op.TablePrinter(fmt, ul='=')(data) )

		fig = plt.figure(figsize=figsize)

		for j in range(len(self.params)):

			for k in range(4):
				iteration = j*4 + k + 1
				ax = fig.add_subplot(len(self.params),4,iteration)

				if iteration in range(1,len(self.params)*4 + 1,4):
					a = sns.distplot(chain[j], rug=False, hist=False)
					a.set_ylabel(self.param_desc[j]['name'])
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
		sns.plt.show()		


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
			if self.model_type == 'GAS':
				plt.figure(figsize=figsize)
				date_index = self.index[max(self.ar,self.sc):len(self.data)]
				mu, Y, scores = self.model(self.params)
				plt.plot(date_index,Y,label='Data')
				plt.plot(date_index,self.link(mu),label='Filter',c='black')
				plt.title(self.data_name)
				plt.legend(loc=2)	
			elif self.model_type == 'VAR':
				date_index = self.index[self.lags:len(self.data[0])]
				mu, Y = self.model(self.params)
				for series in range(len(Y)):
					plt.figure(figsize=figsize)
					plt.plot(date_index,Y[series],label='Data ' + str(series))
					plt.plot(date_index,mu[series],label='Filter' + str(series),c='black')	
					plt.title(self.data_name[series])
					plt.legend(loc=2)	
			elif self.model_type in ['GARCH','EGARCH']:
				plt.figure(figsize=figsize)
				date_index = self.index[max(self.p,self.q):len(self.data)]
				sigma2, Y, ___ = self.model(self.params)
				plt.plot(date_index,np.abs(Y),label=self.data_name + ' Absolute Values')
				if self.model_type == 'GARCH':
					plt.plot(date_index,np.power(sigma2,0.5),label='GARCH(' + str(self.p) + ',' + str(self.q) + ') std',c='black')
				elif self.model_type == 'EGARCH':
					plt.plot(date_index,np.exp(sigma2/2),label='EGARCH(' + str(self.p) + ',' + str(self.q) + ') std',c='black')					
				plt.title(self.data_name + " Volatility Plot")	
				plt.legend(loc=2)	
			elif self.model_type == 'GPNARX':
				self.pfit(self.params)	
			elif self.model_type == 'LLEV':

				date_index = self.index
				mu, _, _ = self.model(self.params)
				mu = mu[0][0:len(mu[0])-1]

				plt.figure(figsize=figsize)	

				plt.subplot(3, 1, 1)
				plt.title(self.data_name + " Raw and Filtered")	

				plt.plot(date_index,self.data,label='Data')
				plt.plot(date_index,mu,label='Filter',c='black')
				plt.legend(loc=2)

				plt.subplot(3, 1, 2)

				plt.title(self.data_name + " Local Level")	
				plt.plot(date_index,mu)

				plt.subplot(3, 1, 3)

				plt.title("Measurement Noise")	
				plt.plot(date_index,self.data-mu)

			elif self.model_type =='LLT':	

				date_index = self.index
				mu, _, _ = self.model(self.params)
				mu0 = mu[0][0:len(mu[0])-1]
				mu1 = mu[1][0:len(mu[1])-1]

				plt.figure(figsize=figsize)	

				plt.subplot(2, 2, 1)
				plt.title(self.data_name + " Raw and Filtered")	

				plt.plot(date_index,self.data,label='Data')
				plt.plot(date_index,mu0,label='Filter',c='black')
				plt.legend(loc=2)

				plt.subplot(2, 2, 2)

				plt.title(self.data_name + " Local Level")	
				plt.plot(date_index,mu0)

				plt.subplot(2, 2, 3)

				plt.title(self.data_name + " Trend")	
				plt.plot(date_index,mu1)

				plt.subplot(2, 2, 4)

				plt.title("Measurement Noise")	
				plt.plot(date_index,self.data-mu0)

			else:
				plt.figure(figsize=figsize)
				date_index = self.index[max(self.ar,self.ma):len(self.data)]
				mu, Y = self.model(self.params)
				plt.plot(date_index,Y,label='Data')
				plt.plot(date_index,mu,label='Filter',c='black')
				plt.title(self.data_name)
				plt.legend(loc=2)	
			plt.show()				

	def bbvi_fit(self,posterior,printer=True,step=0.001,iterations=30000):
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
		phi = np.zeros(self.param_no)

		# Starting values
		if self.model_type == 'GAS':
			if self.dist in ['Laplace','Normal']:
				phi[0] = np.mean(self.data)
			elif self.dist == 'Poisson':
				phi[0] = np.log(np.mean(self.data))
		elif self.model_type == 'VAR':
			phi = self.create_B_direct().flatten()
			cov = self.estimate_cov()

			# Inelegant - needs refactoring
			for i in range(self.ylen):
				for k in range(self.ylen):
					if i == k:
						phi = np.append(phi,self.param_desc[len(phi)]['prior'].itransform(cov[i,k]))
					elif i > k:
						phi = np.append(phi,self.param_desc[len(phi)]['prior'].itransform(cov[i,k]))			
		elif self.model_type == 'GARCH':
			phi = np.ones(self.param_no)*0.00001
			phi[0] = self.param_desc[0]['prior'].itransform(np.mean(np.power(self.data,2)))
		elif self.model_type == 'EGARCH':
			phi[0] = self.param_desc[0]['prior'].itransform(np.log(np.mean(np.power(self.data,2))))
		elif self.model_type == 'GPNARX':
			phi = np.ones(self.param_no)*0.0
			arma_start = arma.ARIMA(self.data,ar=self.ar,ma=0,integ=self.integ)
			arma_start.fit(printer=False)
			phi[0] = arma_start.params[len(arma_start.params)-1]

		self.params = phi

		# Starting values for approximate distribution
		for i in range(len(self.param_desc)):
			approx_dist = self.param_desc[i]['q']
			if isinstance(approx_dist, dst.Normal):
				self.param_desc[i]['q'].loc = self.params[i]
				if len(self.ses) == 0:
					self.param_desc[i]['q'].scale = -3.0
				else:
					self.param_desc[i]['q'].scale = log(self.ses[i])
		q_list = [k['q'] for k in self.param_desc]
		
		bbvi_obj = ifr.BBVI(posterior,q_list,12,step,iterations)
		q, q_params, q_ses = bbvi_obj.lambda_update()
		self.params = q_params

		for k in range(len(self.param_desc)):
			self.param_desc[k]['q'] = q[k]

		self.normal_posterior_sim(self.params,np.diag(np.exp(q_ses)),"Black Box Variational Inference",printer)

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

		date_index = self.index.copy()

		if self.model_type == 'VAR':
			date_index = date_index[self.max_lag:len(self.data[0])]
		else:
			date_index = date_index[self.max_lag:len(self.data)]

		for t in range(h):
			if self.data_type == 'pandas':
				# Only configured for days - need to support smaller time intervals!
				date_index += pd.DateOffset((date_index[len(date_index)-1] - date_index[len(date_index)-2]).days)
			elif self.data_type == 'numpy':
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
			trans_params[k] = self.param_desc[k]['prior'].transform(self.params[k])	
		return trans_params	