from .. import inference as ifr
from .. import output as op
from .. import tests as tst
import numpy as np
import pandas as pd
import scipy.stats as ss
from math import exp, sqrt, log, tanh
import copy
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from .. import covariances as cov
import numdifftools as nd

class ARIMA(object):

	def __init__(self,data,ar,ma,integ=0,target=None):

		# Parameters
		self.ar = ar
		self.ma = ma
		self.int = integ

		# Check Data format
		if isinstance(data, pd.DataFrame):
			self.index = data.index			
			if target is None:
				self.data = data.ix[:,0].values
				self.data_name = data.columns.values[0]
			else:
				self.data = data[target]
				self.index = data.index
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

		# Difference data
		X = self.data
		for order in range(self.int):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		

		# Holding variables for model output
		self.params = []
		self.ses = []
		self.ihessian = []
		self.chains = []

		# Specify parameter description
		self.param_desc = []

		self.param_desc.append({'name' : 'Constant', 'index': 0, 'prior': ifr.Normal(0,3,transform=None)})		
		
		# AR priors
		for j in range(1,self.ar+1):
			self.param_desc.append({'name' : 'AR(' + str(j) + ')', 'index': j, 'prior': ifr.Normal(0,0.5,transform='tanh')})
		
		# MA priors
		for k in range(self.ar+1,self.ar+self.ma+1):
			self.param_desc.append({'name' : 'MA(' + str(k-self.ar) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform='tanh')})
		
		# Variance prior
		self.param_desc.append({'name' : 'Sigma','index': self.ar+self.ma+1, 'prior': ifr.Uniform(transform='exp')})

	# Holds the core model matrices
	def model(self,beta,x):
		Y = np.array(x[max(self.ar,self.ma):len(x)])
		X = np.ones(len(Y))

		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# AR terms
		for i in range(self.ar):
			X = np.vstack((X,x[(max(self.ar,self.ma)-i-1):(len(x)-i-1)]))

		mu = np.matmul(np.transpose(X),parm[0:len(parm)-1-self.ma])

		# MA terms
		if self.ma != 0:
			for t in range(max(self.ar,self.ma),len(Y)):
				for k in range(self.ma):
						mu[t] += parm[1+self.ar+k]*(Y[t-1-k]-mu[t-1-k])

		return mu, Y 

	# Returns negative log likelihood
	def likelihood(self,beta,x):
		mu, Y = self.model(beta,x)
		return -sum(ss.norm.logpdf(mu,loc=Y,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))
		
	# returns unnormalized posterior
	def posterior(self,beta,x):
		post = self.likelihood(beta,x)
		for k in range(self.ar+self.ma+2):
			post += -self.param_desc[k]['prior'].logpdf(beta[k])
		return post

	# Outputs tabular summary of prior assumptions
	def list_priors(self):

		prior_list = []
		prior_desc = []
		param_trans = []

		for i in range(2+self.ar+self.ma):
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


		data = [
		    {'param_index': 0, 'param_name':'Constant', 'param_prior': prior_list[0], 'param_prior_params': prior_desc[0], 'param_trans': param_trans[0]}
		]

		for k in range(self.ar):
			data.append({'param_index': k+1, 'param_name':'AR(' + str(k+1) + ')', 'param_prior': prior_list[k+1], 'param_prior_params': prior_desc[k+1], 'param_trans': param_trans[k+1]})

		for j in range(self.ma):
			data.append({'param_index': j+1+self.ar, 'param_name':'MA(' + str(j+1) + ')', 'param_prior': prior_list[self.ar+j+1], 'param_prior_params': prior_desc[self.ar+j+1], 'param_trans': param_trans[self.ar+j+1]})
		
		data.append({'param_index': self.ar+self.ma+1, 'param_name':'Sigma', 'param_prior': prior_list[self.ar+self.ma+1], 'param_prior_params': prior_desc[self.ar+self.ma+1], 'param_trans': param_trans[self.ar+self.ma+1]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',15),
			('Prior Type','param_prior',10),
			('Prior Parameters', 'param_prior_params', 25),
			('Transformation','param_trans',20)]
		
		print( op.TablePrinter(fmt, ul='=')(data) )

	# Allows for quicker changes to prior specification
	def adjust_prior(self,index,prior):
		if index < 0 or index > (self.ar+self.ma+1) or not isinstance(index, int):
			raise Exception("Oops - the parameter index " + str(index) + " you have entered is invalid!")
		else:
			self.param_desc[index]['prior'] = prior

	# Holding function for inference options
	def fit(self,method,printer=True,nsims=100000,cov_matrix=None):
		if method == 'MLE':
			self.optimize_fit(self.likelihood,printer)
		elif method == 'MAP':
			self.optimize_fit(self.posterior,printer)	
		elif method == 'M-H':
			 self.mcmc_fit(nsims=nsims,method=method,cov_matrix=cov_matrix)
		elif method == "Laplace":
			self.laplace_fit(self.posterior,printer) 
		else:
			raise ValueError("Method not recognized!")

	# Stores point estimates for MAP and MLE estimation; prints summary output
	def optimize_fit(self,obj_type,printer=True):

		# Starting parameters
		phi = np.zeros(self.ar+self.ma+2)

		# Optimize using L-BFGS-B
		p = optimize.minimize(obj_type,phi,args=(self.data),method='L-BFGS-B')
		self.params = p.x

		# Vector for transformed parameters
		t_params = copy.deepcopy(p.x)			

		def lik_wrapper(x):
		    return self.likelihood(x,self.data)

		# Check that matrix is non-singular; act accordingly
		try:
			self.ihessian = np.linalg.inv(nd.Hessian(lik_wrapper)(self.params))
			self.ses = np.diag(self.ihessian)**0.5
			t_p_std = copy.deepcopy(self.ses) # vector for transformed standard errors

			# Create transformed variables
			for k in range(len(t_params)):
				z_temp = (p.x[k]/float(self.ses[k]))
				t_params[k] = self.param_desc[k]['prior'].transform(t_params[k])
				t_p_std[k] = t_params[k] / z_temp

			# Replace with something more elegant in future versions
			if printer is True:

				data = [
				    {'parm_name':'Constant', 'parm_value':round(self.param_desc[0]['prior'].transform(p.x[0]),4), 'parm_std': round(t_p_std[0],4),'parm_z': round(t_params[0]/float(t_p_std[0]),4),'parm_p': round(tst.find_p_value(t_params[0]/float(t_p_std[0])),4),'ci': "(" + str(round(t_params[0] - t_p_std[0]*1.96,4)) + " | " + str(round(t_params[0] + t_p_std[0]*1.96,4)) + ")"}
				]

				for k in range(self.ar):
					data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_value':round(t_params[k+1],4), 'parm_std': round(t_p_std[k+1],4), 
						'parm_z': round(t_params[k+1]/float(t_p_std[k+1]),4), 'parm_p': round(tst.find_p_value(t_params[k+1]/float(t_p_std[k+1])),4),'ci': "(" + str(round(t_params[k+1] - t_p_std[k+1]*1.96,4)) + " | " + str(round(t_params[k+1] + t_p_std[k+1]*1.96,4)) + ")"})

				for k in range(self.ma):
					data.append({'parm_name':'MA(' + str(k+1) + ')', 'parm_value':round(t_params[self.ar+k+1],4), 'parm_std': round(t_p_std[self.ar+k+1],4), 
						'parm_z': round(t_params[self.ar+k+1]/float(t_p_std[self.ar+k+1]),4), 'parm_p': round(tst.find_p_value(t_params[self.ar+k+1]/float(t_p_std[self.ar+k+1])),4),'ci': "(" + str(round(t_params[self.ar+k+1] - t_p_std[self.ar+k+1]*1.96,4)) + " | " + str(round(self.ar+t_params[self.ar+k+1] + t_p_std[self.ar+k+1]*1.96,4)) + ")"})

				data.append({'parm_name':'Sigma', 'parm_value':round(t_params[len(t_params)-1],4), 'parm_std': round(t_p_std[len(t_params)-1],4), 
					'parm_z': round(t_params[len(t_params)-1]/float(t_p_std[len(t_params)-1]),4), 'parm_p': round(tst.find_p_value(t_params[len(t_params)-1]/float(t_p_std[len(t_params)-1])),4),'ci': "(" + str(round(t_params[len(t_params)-1] - t_p_std[len(t_params)-1]*1.96,4)) + " | " + str(round(self.ar+t_params[len(t_params)-1] + t_p_std[len(t_params)-1]*1.96,4)) + ")"})


				fmt = [
				    ('Parameter',       'parm_name',   20),
				    ('Estimate',          'parm_value',       10),
				    ('Standard Error', 'parm_std', 15),
				    ('z',          'parm_z',       10),
				    ('P>|z|',          'parm_p',       10),
				    ('95% Confidence Interval',          'ci',       25)
				]

		except: # If Hessian is not invertible...
			print "Hessian not invertible! Consider a different model specification."
			print ""

			# Transform parameters
			for k in range(len(t_params)):
				t_params[k] = self.param_desc[k]['prior'].transform(t_params[k])

			if printer is True:

				data = [
				    {'parm_name':'Constant', 'parm_value':round(t_params[0],4)}
				]

				for k in range(self.ar):
					data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_value':round(t_params[k+1],4)})

				for k in range(self.sc):
					data.append({'parm_name':'MA(' + str(k+1) + ')', 'parm_value':round(t_params[self.ar+k+1],4)})

				data.append({'parm_name':'Scale', 'parm_value':round(t_params[len(t_params)-1],4)})

				fmt = [
				    ('Parameter',       'parm_name',   20),
				    ('Estimate',          'parm_value',       10)
				]

		# Final printed output
		if printer is True:
			print "ARIMA(" + str(self.ar) + "," + str(self.int) + "," + str(self.ma) + ") regression"
			print "=================="
			if obj_type == self.likelihood:
				print "Method: MLE"
			elif obj_type == self.posterior:
				print "Method: MAP"
			print "Number of observations: " + str(len(self.data)-self.ar)
			print "Log Likelihood: " + str(round(-self.likelihood(p.x,self.data),4))
			if obj_type == self.posterior:
				print "Log Posterior: " + str(round(-self.posterior(p.x,self.data),4))
			print "AIC: " + str(round(2*len(p.x)+2*self.likelihood(p.x,self.data),4))
			print "BIC: " + str(round(2*self.likelihood(p.x,self.data) + len(p.x)*log(len(self.data)-self.ar),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )

	# Performs a Laplace Approximation using MAP point estimates and Inverse Hessian
	def laplace_fit(self,obj_type,printer=True):

		# Get Mode and Inverse Hessian information
		self.fit(method='MAP',printer=False)

		if len(self.ses) == 0: # no errors, no Laplace party
			raise Exception("No Hessian information - Laplace approximation cannot be performed")
		else:

			# Simulate from multivariate normal - silly, use analytical pdfs in future versions
			chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.laplace(self.params,self.ihessian)
			
			# Transform parameters - replace if statement with transform information directly in future version
			for k in range(len(chain)):
				if self.param_desc[k]['prior'].transform == np.exp:
					chain[k] = np.exp(chain[k])
					mean_est[k] = exp(mean_est[k])
					upper_95_est[k] = exp(upper_95_est[k])
					lower_95_est[k] = exp(lower_95_est[k])				
				elif self.param_desc[k]['prior'].transform == np.tanh: 
					chain[k] = np.tanh(chain[k])
					mean_est[k] = tanh(mean_est[k])
					upper_95_est[k] = tanh(upper_95_est[k])
					lower_95_est[k] = tanh(lower_95_est[k])	

		self.chains = chain

		if printer is True:

			data = [
			    {'parm_name':'Constant', 'parm_mean':round(mean_est[0],4), 'parm_median':round(median_est[0],4), 'ci': "(" + str(round(lower_95_est[0],4)) + " | " + str(round(upper_95_est[0],4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_mean':round(mean_est[k+1],4), 'parm_median':round(median_est[k+1],4), 'ci': "(" + str(round(lower_95_est[k+1],4)) + " | " + str(round(upper_95_est[k+1],4)) + ")"})

			for j in range(self.ma):
				data.append({'parm_name':'MA(' + str(j+1) + ')', 'parm_mean':round(mean_est[self.ar+j+1],4), 'parm_median':round(median_est[self.ar+j+1],4), 'ci': "(" + str(round(lower_95_est[j+1+self.ar],4)) + " | " + str(round(upper_95_est[j+1+self.ar],4)) + ")"})

			fmt = [
				('Parameter','parm_name',20),
				('Median','parm_median',10),
				('Mean', 'parm_mean', 15),
				('95% Credibility Interval','ci',25)]
			
			print "ARIMA(" + str(self.ar) + "," + str(self.int) + "," + str(self.ma) + ") regression"
			print "=================="
			print "Method: Laplace Approximation"
			print "Number of observations: " + str(len(self.data)-self.ar)
			print "Log Posterior: " + str(round(-self.posterior(mean_est,self.data),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )
		
		# Plot densities
		for j in range(len(self.params)):
			fig = plt.figure()
			a = sns.distplot(chain[j], rug=False, hist=False, label=self.param_desc[j]['name'])
			a.set_title(self.param_desc[j]['name'])
			a.set_ylabel('Density')
				
		sns.plt.show()		

	# Wrapper function for MCMC inference methods
	def mcmc_fit(self,scale=(2.38/sqrt(10000)),nsims=100000,printer=True,method="M-H",cov_matrix=None):

		# Initialize with MAP estimates
		self.fit(method='MAP',printer=False)
		
		if method == "M-H":
			chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.metropolis_hastings(self.data,self.posterior,scale,nsims,self.params,cov_matrix=cov_matrix)
		else:
			raise Exception("Method not recognized!")

		self.params = np.asarray(mean_est)

		# Transform parameters - replace if statement with transform information directly in future version
		for k in range(len(chain)):
			if self.param_desc[k]['prior'].transform == exp:
				chain[k] = np.exp(chain[k])
				mean_est[k] = exp(mean_est[k])
				upper_95_est[k] = exp(upper_95_est[k])
				lower_95_est[k] = exp(lower_95_est[k])				
			elif self.param_desc[k]['prior'].transform == tanh: 
				chain[k] = np.tanh(chain[k])
				mean_est[k] = tanh(mean_est[k])
				upper_95_est[k] = tanh(upper_95_est[k])
				lower_95_est[k] = tanh(lower_95_est[k])	

		self.chains = chain

		if printer is True:

			data = [
			    {'parm_name':'Constant', 'parm_mean':round(mean_est[0],4), 'parm_median':round(median_est[0],4), 'ci': "(" + str(round(lower_95_est[0],4)) + " | " + str(round(upper_95_est[0],4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_mean':round(mean_est[k+1],4), 'parm_median':round(median_est[k+1],4), 'ci': "(" + str(round(lower_95_est[k+1],4)) + " | " + str(round(upper_95_est[k+1],4)) + ")"})

			for j in range(self.ma):
				data.append({'parm_name':'MA(' + str(j+1) + ')', 'parm_mean':round(mean_est[self.ar+j+1],4), 'parm_median':round(median_est[self.ar+j+1],4), 'ci': "(" + str(round(lower_95_est[j+1+self.ar],4)) + " | " + str(round(upper_95_est[j+1+self.ar],4)) + ")"})

		fmt = [
			('Parameter','parm_name',20),
			('Median','parm_median',10),
			('Mean', 'parm_mean', 15),
			('95% Credibility Interval','ci',25)]
		
		print "ARIMA(" + str(self.ar) + "," + str(self.int) + "," + str(self.ma) + ") regression"
		print "=================="
		print "Method: Metropolis-Hastings"
		print "Number of simulations: " + str(nsims)
		print "Number of observations: " + str(len(self.data)-self.ar)
		print "Log Posterior: " + str(round(-self.posterior(mean_est,self.data),4))
		print ""
		print( op.TablePrinter(fmt, ul='=')(data) )

		# Construct MCMC summary plot
		fig = plt.figure()

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

	# Produces T-step ahead forecast for the series
	# This code is very inefficient; needs amending
	def predict(self,T=5,lookback=20,intervals=True):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Get data
			exp_values = self.data[max(self.ar,self.ma):len(self.data)]
			date_index = self.index[max(self.ar,self.ma):len(self.data)]
			
			# Create future dates
			for t in range(T):
				if self.data_type == 'pandas':
					date_index += pd.DateOffset(1)
				elif self.data_type == 'numpy':
					date_index.append(date_index[len(date_index)-1]+1)

			# Grab implied model information using estimated parameters
			mu, Y = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Expected values of the series
			mu_exp = mu
			for t in range(T):
				new_value = params_to_use[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += params_to_use[j]*exp_values[len(exp_values)-1-j]

				if self.ma != 0:
					for k in range(1,self.ma+1):
						if (k-1) >= t:
							new_value += params_to_use[k+self.ar]*(exp_values[len(exp_values)-1-k]-mu_exp[len(exp_values)-1-k])

				exp_values = np.append(exp_values,[new_value])
				mu_exp = np.append(mu_exp,[0]) # For indexing consistency

			# Simulate error bars (do analytically in future)
			sim_vector = np.zeros([10000,T])

			for n in range(10000):
				mu_exp = mu
				values = self.data[max(self.ar,self.ma):len(self.data)]
				for t in range(T):
					new_value = params_to_use[0] + np.random.randn(1)*params_to_use[len(params_to_use)-1]

					if self.ar != 0:
						for j in range(1,self.ar+1):
							new_value += params_to_use[j]*values[len(values)-1-j]

					if self.ma != 0:
						for k in range(1,self.ma+1):
							if (k-1) >= t:
								new_value += params_to_use[k+self.ar]*(values[len(values)-1-k]-mu_exp[len(values)-1-k])

					values = np.append(values,[new_value])
					mu_exp = np.append(mu_exp,[0]) # For indexing consistency

				sim_vector[n] = values[(len(values)-T):(len(values))]

			test = np.transpose(sim_vector)
			error_bars = [np.percentile(i,95) for i in test] - exp_values[(len(values)-T):(len(values))]
			error_bars_90 = [np.percentile(i,90) for i in test] - exp_values[(len(values)-T):(len(values))]
			forecasted_values = exp_values[(len(values)-T):(len(values))]
			exp_values = exp_values[len(exp_values)-T-lookback:len(exp_values)]
			date_index = date_index[len(date_index)-T-lookback:len(date_index)]
			
			# Plot prediction graph
			if intervals is True:
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)			
			plt.plot(date_index,exp_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()





