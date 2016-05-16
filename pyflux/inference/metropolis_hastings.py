import numpy as np
from math import exp
import sys
if sys.version_info < (3,):
    range = xrange

from scipy.stats import multivariate_normal

class MetropolisHastings(object):
	""" RANDOM-WALK METROPOLIS-HASTINGS MCMC

	Parameters
	----------
	posterior : function
		A posterior function

	scale : float
		The scale for the random walk

	nsims : int
		The number of simulations to perform

	initials : np.array
		Where to start the MCMC chain

	cov_matrix : np.array
		(optional) A covariance matrix for the random walk
	
	model_object : TSM object
		A model object (for use in SPDK sampling)
	"""

	def __init__(self,posterior,scale,nsims,initials,cov_matrix=None,model_object=None):
		self.posterior = posterior
		self.scale = scale
		self.nsims = nsims
		self.initials = initials
		self.param_no = self.initials.shape[0]
		self.phi = np.zeros([nsims,self.param_no])
		self.phi[0] = self.initials
		if cov_matrix is None:
			self.cov_matrix = np.identity(self.param_no)
		else:
			self.cov_matrix = cov_matrix

		if model_object is not None:
			self.model = model_object

	@staticmethod
	def tune_scale(acceptance,scale):
		""" Tunes scale for M-H algorithm

		Parameters
		----------
		acceptance : float
			The most recent acceptance rate

		scale : float
			The current scale parameter

		Returns
		----------
		scale : float
			An adjusted scale parameter
		"""		

		if acceptance > 0.8:
			scale *= 2.0
		elif acceptance <= 0.8 and acceptance > 0.4:
			scale *= 1.3			
		elif acceptance < 0.234 and acceptance > 0.1:
			scale *= (1/1.3)
		elif acceptance <= 0.1 and acceptance > 0.05:
			scale *= 0.4
		elif acceptance <= 0.05 and acceptance > 0.01:
			scale *= 0.2
		elif acceptance <= 0.01:
			scale *= 0.05
		return scale		

	def sample(self):
		""" Samples from posterior (hopefully!)

		Returns
		----------
		chain : np.array
			Chains for each parameter - warning, can use up a lot of memory

		mean_est : np.array
			Mean values for each parameter

		median_est : np.array
			Median values for each parameter

		upper_95_est : np.array
			Upper 95% credibility interval for each parameter

		lower_95_est : np.array
			Lower 95% credibility interval for each parameter			
		"""		

		acceptance = 1
		finish = 0

		while (acceptance < 0.234 or acceptance > 0.4) or finish == 0:

			# If acceptance is in range, proceed to sample, else continue tuning
			if not (acceptance < 0.234 or acceptance > 0.4):
				finish = 1
				print("")
				print("Tuning complete! Now sampling.")
				sims_to_do = self.nsims
			else:
				sims_to_do = 5000 # For acceptance rate tuning

			# Holds data on acceptance rates and uniform random numbers
			a_rate = np.zeros([sims_to_do,1])
			crit = np.random.rand(sims_to_do,1)
			post = multivariate_normal(np.zeros(self.param_no),self.cov_matrix)
			rnums = post.rvs()*self.scale
			for k in range(1,sims_to_do): 
				rnums = np.vstack((rnums,post.rvs()*self.scale))

			old_lik = -self.posterior(self.phi[0]) # Initial posterior

			# Sampling time!
			for i in range(1,sims_to_do):
				phi_prop = self.phi[i-1] + rnums[i]
				post_prop = -self.posterior(phi_prop)
				lik_rat = exp(post_prop - old_lik)

				if crit[i] < lik_rat:
					self.phi[i] = phi_prop
					a_rate[i] = 1
					old_lik = post_prop
				else:
					self.phi[i] = self.phi[i-1]

			acceptance = a_rate.sum()/a_rate.shape[0]
			self.scale = self.tune_scale(acceptance,self.scale)

			print("Acceptance rate of Metropolis-Hastings is " + str(acceptance))

		chain = np.array([self.phi[i][0] for i in range(0,self.phi.shape[0])])
		for m in range(1,self.param_no):
			chain = np.vstack((chain,[self.phi[i][m] for i in range(0,self.phi.shape[0])]))

		mean_est = np.array([np.mean(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
		median_est = np.array([np.median(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
		upper_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]),95) for j in range(self.param_no)])
		lower_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]),5) for j in range(self.param_no)])

		return chain, mean_est, median_est, upper_95_est, lower_95_est

	def spdk_sample(self,smoother_weight):
		""" Samples from SSM posterior using SPDK simulation smoothing

		Parameters
		----------

		smoother_weight : float
			How much weight to give to simulation smoother Samples
			(Default = 0.1). If acceptance rate tuning proves ineffective/slow,
			then decrease the simulation smoother weight.

		Returns
		----------
		chain : np.array
			Chains for each parameter - warning, can use up a lot of memory

		mean_est : np.array
			Mean values for each parameter

		median_est : np.array
			Median values for each parameter

		upper_95_est : np.array
			Upper 95% credibility interval for each parameter

		lower_95_est : np.array
			Lower 95% credibility interval for each parameter			
		"""

		acceptance = 1
		finish = 0

		states = np.zeros([self.nsims, self.model.state_no, self.model.data.shape[0]])
		T_start, Z_start, R_start, Q_start = self.model._ss_matrices(self.phi[0])
		H_start, mu_start = self.model._approximating_model(self.phi[0],T_start,Z_start,R_start,Q_start)

		# Find starting states
		states_start = 0
		states[0,:,:] = self.model.simulation_smoother(self.phi[0])

		while (acceptance < 0.234 or acceptance > 0.4) or finish == 0:

			# If acceptance is in range, proceed to sample, else continue tuning
			if not (acceptance < 0.234 or acceptance > 0.4):
				finish = 1
				print("")
				print("Tuning and warm-up complete! Now sampling.")
				states[0,:,:] = states[round(self.nsims/2.0)-1,:,:]
				self.phi[0] = self.phi[round(self.nsims/2.0)-1]
				sims_to_do = self.nsims
			else:
				sims_to_do = int(round(self.nsims/2.0)) # For acceptance rate tuning
				if acceptance != 1:
					states[0,:,:] = states[round(self.nsims/2.0)-1,:,:]
					self.phi[0] = self.phi[round(self.nsims/2.0)-1]

			# Holds data on acceptance rates and uniform random numbers
			a_rate = np.zeros([sims_to_do,1])
			crit = np.random.rand(sims_to_do,1)
			post = multivariate_normal(np.zeros(self.param_no),self.cov_matrix)
			rnums = post.rvs()*self.scale
			for k in range(1,sims_to_do): 
				rnums = np.vstack((rnums,post.rvs()*self.scale))
			old_lik = -self.posterior(self.phi[0],states[0,:,:]) # Initial posterior

			# Sampling time!
			for i in range(1,sims_to_do):
				phi_prop = self.phi[i-1] + rnums[i]
				states_prop = smoother_weight*self.model.simulation_smoother(phi_prop) + (1-smoother_weight)*states[i-1,:,:]			
				prop_post = -self.posterior(phi_prop,states_prop)
				lik_rat = np.exp(prop_post - old_lik)

				if crit[i] < lik_rat:
					self.phi[i] = phi_prop
					states[i,:,:] = states_prop
					a_rate[i] = 1
					old_lik = prop_post
				else:
					self.phi[i] = self.phi[i-1]
					states[i,:,:] = states[i-1,:,:]

			acceptance = a_rate.sum()/a_rate.shape[0]
			self.scale = self.tune_scale(acceptance,self.scale)

			print("Acceptance rate of Metropolis-Hastings is " + str(acceptance))

		for state_number in range(self.model.state_no):
			states_vector = states[:,state_number,:]

			if state_number == 0:
				state_mean = np.mean(states_vector,axis=0)
				state_median = np.median(states_vector,axis=0)
				state_upper_95 = np.percentile(states_vector,95,axis=0)
				state_lower_95 = np.percentile(states_vector,5,axis=0)
			else:
				state_mean = np.array([state_mean,np.mean(states_vector,axis=0)])
				state_median = np.array([state_median,np.median(states_vector,axis=0)])
				state_upper_95 = np.array([state_upper_95,np.percentile(states_vector,95,axis=0)])
				state_lower_95 = np.array([state_lower_95,np.percentile(states_vector,5,axis=0)])			

		chain = np.array([self.phi[i][0] for i in range(0,self.phi.shape[0])])
		for m in range(1,self.param_no):
			chain = np.vstack((chain,[self.phi[i][m] for i in range(0,self.phi.shape[0])]))

		mean_est = np.array([np.mean(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
		median_est = np.array([np.median(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
		upper_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]),95) for j in range(self.param_no)])
		lower_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]),5) for j in range(self.param_no)])

		return chain, mean_est, median_est, upper_95_est, lower_95_est, states, state_mean, state_median, state_upper_95, state_lower_95