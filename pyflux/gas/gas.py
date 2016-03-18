from .. import inference as ifr
from .. import output as op
from .. import tests as tst
import numpy as np
import pandas as pd
import scipy.stats as ss
import copy
from scores import *
from math import exp, sqrt, log, tanh
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from .. import covariances as cov

class GAS(object):

	def __init__(self,data,dist,ar,sc,integ=0,target=None):

		# Parameters

		self.dist = dist
		self.ar = ar
		self.sc = sc
		self.int = integ

		# Check pandas or numpy

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
			raise Exception("The data input is not pandas or numpy compatible!")

		# Difference data
		X = self.data
		for order in range(self.int):
			X = np.diff(X)
			self.data_name = "Differenced " + self.data_name
		self.data = X		

		# Holders for fitted models
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
		for k in range(self.ar+1,self.ar+self.sc+1):
			self.param_desc.append({'name' : 'SC(' + str(k-self.sc) + ')', 'index': k, 'prior': ifr.Normal(0,0.5,transform='tanh')})
		# Variance prior
		self.param_desc.append({'name' : 'Scale','index': self.ar+self.sc+1, 'prior': ifr.Uniform(transform='exp')})

	def model(self,beta,x):
		Y = np.array(x[max(self.ar,self.sc):len(x)])
		theta = np.ones(len(Y))*beta[0]
		scores = np.zeros(len(Y))
 
		# Transform parameters
		parm = [self.param_desc[k]['prior'].transform(beta[k]) for k in range(len(beta))]

		# Loop over time series
		for t in range(max(self.ar,self.sc),len(Y)):
			# BUILD MEAN PREDICTION
			theta[t] = parm[0]

			# Loop over AR terms
			for ar_term in range(self.ar):
				theta[t] += parm[1+ar_term]*theta[t-ar_term-1]

			# Loop over Score terms
			for sc_term in range(self.sc):
				theta[t] += parm[1+self.ar+sc_term]*scores[t-sc_term-1]

			scores[t] = lik_score(Y[t],theta[t],parm[len(parm)-1],self.dist)

		return theta, Y, scores

	def likelihood(self,beta,x):
		theta, Y, scores = self.model(beta,x)

		if self.dist == "Laplace":
			return -sum(ss.laplace.logpdf(theta,loc=Y,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))
		elif self.dist == "Normal":
			return sum(ss.norm.pdf(theta,loc=Y,scale=self.param_desc[len(beta)-1]['prior'].transform(beta[len(beta)-1])))	

	def adjust_prior(self,index,prior):
		if index < 0 or index > (self.ar+self.sc+1) or not isinstance(index, int):
			raise Exception("Oops - the parameter index " + str(index) + " you have entered is invalid!")
		else:
			self.param_desc[index]['prior'] = prior
		
	def posterior(self,beta,x):
		post = self.likelihood(beta,x)
		for k in range(self.ar+self.sc+2):
			post += -self.param_desc[k]['prior'].logpdf(beta[k])
		return post

	def list_priors(self):

		prior_list = []
		prior_desc = []
		param_trans = []

		for i in range(2+self.ar+self.sc):
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
				prior_desc.append('n/a (uninformative)')
			else:
				raise Exception("Error - prior distribution not detected!")


		data = [
		    {'param_index': 0, 'param_name':'Constant', 'param_prior': prior_list[0], 'param_prior_params': prior_desc[0], 'param_trans': param_trans[0]}
		]

		for k in range(self.ar):
			data.append({'param_index': k+1, 'param_name':'AR(' + str(k+1) + ')', 'param_prior': prior_list[k+1], 'param_prior_params': prior_desc[k+1], 'param_trans': param_trans[k+1]})

		for j in range(self.sc):
			data.append({'param_index': j+1+self.ar, 'param_name':'SC(' + str(j+1) + ')', 'param_prior': prior_list[self.ar+j+1], 'param_prior_params': prior_desc[self.ar+j+1], 'param_trans': param_trans[self.ar+j+1]})
		
		data.append({'param_index': self.ar+self.sc+1, 'param_name':'Scale', 'param_prior': prior_list[self.ar+self.sc+1], 'param_prior_params': prior_desc[self.ar+self.sc+1], 'param_trans': param_trans[self.ar+self.sc+1]})

		fmt = [
			('Index','param_index',6),		
			('Parameter','param_name',15),
			('Prior Type','param_prior',10),
			('Prior Parameters', 'param_prior_params', 25),
			('Transformation','param_trans',20)]
		
		print( op.TablePrinter(fmt, ul='=')(data) )


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
			raise Exception("Method not recognized!")

	def optimize_fit(self,obj_type,printer=True):
		# Difference dependent variable
		X = self.data

		phi = np.zeros(self.ar+self.sc+2)

		# Use L-BFGS-B to find a decent solution; then use BFGS to get final solution and iHessian
		p = optimize.minimize(obj_type,phi,args=(X),method='L-BFGS-B')
		p = optimize.minimize(obj_type,p.x,args=(X),method='BFGS')

		self.params = p.x
		p_std = np.diag(p.hess_inv)**0.5
		self.ses = p_std
		self.ihessian = p.hess_inv

		# Transform parameters
		t_params = copy.deepcopy(p.x)
		t_p_std = copy.deepcopy(p_std)

		# In future versions -> approximate Hessian when this happens
		if sum(p_std) == len(p_std):
			if printer==True:
				print "WARNING: Hessian not estimated! Use alternative fitting method to get uncertainty estimates, e.g. MCMC"
			self.params = np.zeros(len(self.params))
			for k in range(len(p_std)):
				t_params[k] = self.param_desc[k]['prior'].transform(p.x[k])
		else:
			for k in range(len(p_std)):
				z_temp = (p.x[k]/float(p_std[k]))
				t_params[k] = self.param_desc[k]['prior'].transform(p.x[k])
				t_p_std[k] = t_params[k] / z_temp

		# Format parameters

		if printer == True:

			data = [
			    {'parm_name':'Constant', 'parm_value':round(self.param_desc[0]['prior'].transform(p.x[0]),4), 'parm_std': round(p_std[0],4),'parm_z': round(p.x[0]/float(p_std[0]),4),'parm_p': round(tst.find_p_value(p.x[0]/float(p_std[0])),4),'ci': "(" + str(round(p.x[0] - p_std[0]*1.96,4)) + " | " + str(round(p.x[0] + p_std[0]*1.96,4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_value':round(t_params[k+1],4), 'parm_std': round(t_p_std[k+1],4), 
					'parm_z': round(t_params[k+1]/float(t_p_std[k+1]),4), 'parm_p': round(tst.find_p_value(t_params[k+1]/float(t_p_std[k+1])),4),'ci': "(" + str(round(t_params[k+1] - t_p_std[k+1]*1.96,4)) + " | " + str(round(t_params[k+1] + t_p_std[k+1]*1.96,4)) + ")"})

			for k in range(self.sc):
				data.append({'parm_name':'SC(' + str(k+1) + ')', 'parm_value':round(t_params[self.ar+k+1],4), 'parm_std': round(t_p_std[self.ar+k+1],4), 
					'parm_z': round(t_params[self.ar+k+1]/float(t_p_std[self.ar+k+1]),4), 'parm_p': round(tst.find_p_value(t_params[self.ar+k+1]/float(t_p_std[self.ar+k+1])),4),'ci': "(" + str(round(t_params[self.ar+k+1] - t_p_std[self.ar+k+1]*1.96,4)) + " | " + str(round(self.ar+t_params[k+1] + t_p_std[self.ar+k+1]*1.96,4)) + ")"})

			fmt = [
			    ('Parameter',       'parm_name',   20),
			    ('Estimate',          'parm_value',       10),
			    ('Standard Error', 'parm_std', 15),
			    ('z',          'parm_z',       10),
			    ('P>|z|',          'parm_p',       10),
			    ('95% Confidence Interval',          'ci',       25)
			]

			print "GAS(" + str(self.ar) + "," + str(self.int) + "," + str(self.sc) + ") regression"
			print "=================="
			if obj_type == self.likelihood:
				print "Method: MLE"
			elif obj_type == self.posterior:
				print "Method: MAP"
			print "Number of observations: " + str(len(X)-self.ar)
			print "Log Likelihood: " + str(round(-self.likelihood(p.x,X),4))
			if obj_type == self.posterior:
				print "Log Posterior: " + str(round(-self.posterior(p.x,X),4))
			print "AIC: " + str(round(2*len(p.x)+2*self.likelihood(p.x,X),4))
			print "BIC: " + str(round(2*self.likelihood(p.x,X) + len(p.x)*log(len(X)-self.ar),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )

	def laplace_fit(self,obj_type,printer=True):

		self.fit(method='MAP',printer=False)
		chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.laplace(self.params,self.ihessian)
		
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

		if printer == True:

			data = [
			    {'parm_name':'Constant', 'parm_mean':round(mean_est[0],4), 'parm_median':round(median_est[0],4), 'ci': "(" + str(round(lower_95_est[0],4)) + " | " + str(round(upper_95_est[0],4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_mean':round(mean_est[k+1],4), 'parm_median':round(median_est[k+1],4), 'ci': "(" + str(round(lower_95_est[k+1],4)) + " | " + str(round(upper_95_est[k+1],4)) + ")"})

			for j in range(self.sc):
				data.append({'parm_name':'SC(' + str(j+1) + ')', 'parm_mean':round(mean_est[self.ar+j+1],4), 'parm_median':round(median_est[self.ar+j+1],4), 'ci': "(" + str(round(lower_95_est[j+1+self.ar],4)) + " | " + str(round(upper_95_est[j+1+self.ar],4)) + ")"})

		fmt = [
			('Parameter','parm_name',20),
			('Median','parm_median',10),
			('Mean', 'parm_mean', 15),
			('95% Credibility Interval','ci',25)]
		
		print "GAS(" + str(self.ar) + "," + str(self.int) + "," + str(self.sc) + ") regression"
		print "=================="
		print "Method: Laplace Approximation"
		print "Number of observations: " + str(len(self.data)-self.ar)
		print "Log Posterior: " + str(round(-self.posterior(mean_est,self.data),4))
		print ""
		print( op.TablePrinter(fmt, ul='=')(data) )

		fig = plt.figure()

		for j in range(len(self.params)):
			fig = plt.figure()
			a = sns.distplot(chain[j], rug=False, hist=False, label=self.param_desc[j]['name'])
			a.set_title(self.param_desc[j]['name'])
			a.set_ylabel('Density')
				
		sns.plt.show()			

	def mcmc_fit(self,scale=(2.38/sqrt(10000)),nsims=100000,printer=True,method="M-H",cov_matrix=None):

		X = self.data
		self.fit(method='MAP',printer=False)
		
		if method == "M-H":
			chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.metropolis_hastings(X,self.posterior,scale,nsims,self.params,cov_matrix=cov_matrix)
		elif method == "HMC":
			def posterior_c(phi):
				return self.posterior(phi,X)
			chain, mean_est, median_est, upper_95_est, lower_95_est = ifr.hmc(X,posterior_c,nsims,self.params,self.ses)
		else:
			raise Exception("Method not recognized!")

		self.params = mean_est

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

		if printer == True:

			data = [
			    {'parm_name':'Constant', 'parm_mean':round(mean_est[0],4), 'parm_median':round(median_est[0],4), 'ci': "(" + str(round(lower_95_est[0],4)) + " | " + str(round(upper_95_est[0],4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_mean':round(mean_est[k+1],4), 'parm_median':round(median_est[k+1],4), 'ci': "(" + str(round(lower_95_est[k+1],4)) + " | " + str(round(upper_95_est[k+1],4)) + ")"})

			for j in range(self.sc):
				data.append({'parm_name':'SC(' + str(j+1) + ')', 'parm_mean':round(mean_est[self.ar+j+1],4), 'parm_median':round(median_est[self.ar+j+1],4), 'ci': "(" + str(round(lower_95_est[j+1+self.ar],4)) + " | " + str(round(upper_95_est[j+1+self.ar],4)) + ")"})

		fmt = [
			('Parameter','parm_name',20),
			('Median','parm_median',10),
			('Mean', 'parm_mean', 15),
			('95% Credibility Interval','ci',25)]
		
		print "GAS(" + str(self.ar) + "," + str(self.int) + "," + str(self.sc) + ") regression"
		print "=================="
		print "Method: Metropolis-Hastings"
		print "Number of simulations: " + str(nsims)
		print "Number of observations: " + str(len(X)-self.ar)
		print "Log Posterior: " + str(round(-self.posterior(mean_est,X),4))
		print ""
		print( op.TablePrinter(fmt, ul='=')(data) )

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


	def predict(self,T=5,lookback=20,intervals=True):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Get data
			exp_values = self.data[max(self.ar,self.sc):len(self.data)]
			date_index = self.index[max(self.ar,self.sc):len(self.data)]
			
			for t in range(T):
				if self.data_type == 'pandas':
					date_index += pd.DateOffset(1)
				elif self.data_type == 'numpy':
					date_index.append(date_index[len(date_index)-1]+1)

			mu, Y, scores = self.model(self.params,self.data)

			# Get and format parameters
			params_to_use = copy.deepcopy(self.params)
			for k in range(len(self.params)):
				params_to_use[k] = self.param_desc[k]['prior'].transform(self.params[k])

			# Expectation
			mu_exp = mu
			for t in range(T):
				new_value = params_to_use[0]

				if self.ar != 0:
					for j in range(1,self.ar+1):
						new_value += params_to_use[j]*exp_values[len(exp_values)-1-j]

				if self.sc != 0:
					for k in range(1,self.sc+1):
						new_value += params_to_use[k+self.ar]*scores[len(exp_values)-1-k]

				exp_values = np.append(exp_values,[new_value])
				mu_exp = np.append(mu_exp,[0]) # For indexing consistency
				scores = np.append(scores,[0]) # expectation of score is zero

			# Simulate error bars (do analytically in future)
			# Expectation
			sim_vector = np.zeros([10000,T])

			for n in range(10000):
				mu_exp = mu
				values = self.data[max(self.ar,self.sc):len(self.data)]
				for t in range(T):
					new_value = params_to_use[0]

					if self.ar != 0:
						for j in range(1,self.ar+1):
							new_value += params_to_use[j]*exp_values[len(exp_values)-1-j]

					if self.sc != 0:
						for k in range(1,self.sc+1):
							new_value += params_to_use[k+self.ar]*scores[len(exp_values)-1-k]

					if self.dist == "Normal":
						new_value = np.random.normal(new_value, params_to_use[len(params_to_use)-1], 1)[0]
					elif self.dist == "Laplace":
						new_value = np.random.laplace(new_value, params_to_use[len(params_to_use)-1], 1)[0]

					values = np.append(values,[new_value])
					mu_exp = np.append(mu_exp,[0]) # For indexing consistency

				sim_vector[n] = values[(len(values)-T):(len(values))]

			test = np.transpose(sim_vector)
			error_bars = [np.percentile(i,95) for i in test] - exp_values[(len(values)-T):(len(values))]
			error_bars_90 = [np.percentile(i,90) for i in test] - exp_values[(len(values)-T):(len(values))]
			forecasted_values = exp_values[(len(values)-T):(len(values))]
			exp_values = exp_values[len(exp_values)-T-lookback:len(exp_values)]
			date_index = date_index[len(date_index)-T-lookback:len(date_index)]
			
			print exp_values

			if intervals == True:
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars_90, forecasted_values+error_bars_90,alpha=0.25)
				plt.fill_between(date_index[len(date_index)-T:len(date_index)], forecasted_values-error_bars, forecasted_values+error_bars,alpha=0.5)			
			plt.plot(date_index,exp_values)
			plt.title("Forecast for " + self.data_name)
			plt.xlabel("Time")
			plt.ylabel(self.data_name)
			plt.show()
