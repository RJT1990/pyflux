import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as ss

class BBVI(object):

	def __init__(self,neg_posterior,q,sims,step=0.001,iterations=30000):
		self.neg_posterior = neg_posterior
		self.q = q
		self.sims = sims
		self.iterations = iterations
		self.approx_param_no = np.array([i.param_no for i in self.q])
		self.step = step
		self.printer = True

	def get_means_and_scales(self):
		means = np.zeros(len(self.q))
		scale = np.zeros(len(self.q))
		for i in range(len(self.q)):
			means[i] = self.q[i].loc
			scale[i] = np.exp(self.q[i].scale)		
		return means, scale

	def draw_variables(self):
		z = self.q[0].sim(self.sims)
		for i in range(1,len(self.q)):
			z = np.vstack((z,self.q[i].sim(self.sims)))
		return z

	def draw_normal(self):
		means, scale = self.get_means_and_scales()
		return np.random.normal(means,scale,size=[self.sims,len(means)]).T

	def create_normal_logq(self,z):
		means, scale = self.get_means_and_scales()
		return ss.norm.logpdf(z,loc=means,scale=scale).sum()

	def grad_log_q(self,z):
		param_count = 0
		grad = np.zeros((np.sum(self.approx_param_no),self.sims))
		for core_param in range(len(self.q)):
			for approx_param in range(self.q[core_param].param_no):
				grad[param_count] = self.q[core_param].score(z[core_param],approx_param)		
				param_count += 1
		return grad

	def log_p(self,z):
		z_t = np.transpose(z)
		return np.array([-self.neg_posterior(i) for i in z_t])

	def normal_log_q(self,z):
		z_t = np.transpose(z)
		means, scale = self.get_means_and_scales()
		return (ss.norm.logpdf(z_t,loc=means,scale=scale)).sum(axis=1)

	def cv_gradient(self,z):
		gradient = np.zeros(np.sum(self.approx_param_no))
		log_q = self.normal_log_q(z)
		log_p = self.log_p(z)
		grad_log_q = self.grad_log_q(z)
		for lambda_i in range(np.sum(self.approx_param_no)):
			alpha = np.cov(grad_log_q[lambda_i],grad_log_q[lambda_i]*(log_p-log_q))[0][1]/np.var(grad_log_q[lambda_i])				
			gradient[lambda_i] = np.mean(grad_log_q[lambda_i]*(log_p-log_q) - alpha*grad_log_q[lambda_i])
		return gradient

	def current_parameters(self):
		current = []
		for core_param in range(len(self.q)):
			for approx_param in range(self.q[core_param].param_no):
				current.append(self.q[core_param].return_param(approx_param))
		return np.array(current)

	def change_parameters(self,params):
		no_of_params = 0
		for core_param in range(len(self.q)):
			for approx_param in range(self.q[core_param].param_no):
				self.q[core_param].change_param(approx_param,params[no_of_params])
				no_of_params += 1

	def return_elbo(self,params):
		self.change_parameters(params)
		return np.mean(np.asarray([self.neg_posterior(i) for i in np.transpose(self.draw_variables())]))

	def print_progress(self,i,current_params):
		for split in range(1,11):
			if i == (round(self.iterations/10*split)-1):
				print(str(split) + "0% done : ELBO is " + str(-self.neg_posterior(current_params)-self.create_normal_logq(current_params)))

	def lambda_update(self):
		Gjj = 0
		final_parameters = self.current_parameters()
		final_samples = 1
		for i in range(self.iterations):
			
			# Draw variables and gradients
			z = self.draw_normal()
			gradient = self.cv_gradient(z)
			gradient[np.isnan(gradient)] = 0

			# RMS prop
			Gjj = 0.99*Gjj + 0.01*np.power(gradient,2)
			new_parameters = self.current_parameters() + self.step*(gradient/np.sqrt(Gjj))	
			self.change_parameters(new_parameters)

			# Print progress
			current_z = self.current_parameters()
			current_lambda = np.array([current_z[el] for el in range(len(current_z)) if el%2==0])		
			
			if self.printer is True:
				self.print_progress(i,current_lambda)

			# Construct final parameters using final 10% of samples
			if i > self.iterations-round(self.iterations/10):
				final_samples += 1
				final_parameters = final_parameters+current_z

		final_parameters = final_parameters/float(final_samples)
		final_means = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2==0])
		final_ses = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2!=0])
		if self.printer is True:
			print("")
			print("Final model ELBO is " + str(-self.neg_posterior(final_means)-self.create_normal_logq(final_means)))
		return self.q, final_means, final_ses

	def lambda_update_store(self):
		"""
		Stores the history of parameter updates
		"""

		Gjj = 0
		final_parameters = self.current_parameters()
		final_samples = 1

		stored_means = np.zeros((self.iterations,len(final_parameters)/2))
		stored_predictive_likelihood = np.zeros(self.iterations)
		for i in range(self.iterations):
			
			# Draw variables and gradients
			z = self.draw_normal()
			gradient = self.cv_gradient(z)
			gradient[np.isnan(gradient)] = 0

			# RMS prop
			Gjj = 0.99*Gjj + 0.01*np.power(gradient,2)
			new_parameters = self.current_parameters() + self.step*(gradient/np.sqrt(Gjj))	
			self.change_parameters(new_parameters)
			stored_means[i] = np.array([new_parameters[el] for el in range(len(new_parameters)) if el%2==0])
			stored_predictive_likelihood[i] = self.neg_posterior(stored_means[i])

			# Print progress
			current_z = self.current_parameters()
			current_lambda = np.array([current_z[el] for el in range(len(current_z)) if el%2==0])		
			
			if self.printer is True:
				self.print_progress(i,current_lambda)

			# Construct final parameters using final 10% of samples
			if i > self.iterations-round(self.iterations/10):
				final_samples += 1
				final_parameters = final_parameters+current_z

		final_parameters = final_parameters/float(final_samples)
		final_means = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2==0])
		final_ses = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2!=0])

		if self.printer is True:
			print("")
			print("Final model ELBO is " + str(-self.neg_posterior(final_means)-self.create_normal_logq(final_means)))
		return self.q, final_means, final_ses, stored_means, stored_predictive_likelihood
