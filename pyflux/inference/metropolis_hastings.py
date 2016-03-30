import numpy as np
from math import exp
from scipy.stats import multivariate_normal

def metropolis_hastings(posterior,scale,nsims,initials,cov_matrix=None):
	phi = np.zeros([nsims,len(initials)])
	phi[0] = initials

	if cov_matrix is None:
		cov_matrix = np.identity(len(initials))

	# Initialize parameters for while loop
	acceptance = 1
	finish = 0

	while (acceptance < 0.234 or acceptance > 0.4) or finish == 0:
		
		# If acceptance is in range, proceed to sample, else continue tuning
		if not (acceptance < 0.234 or acceptance > 0.4):
			finish = 1
			print ""
			print "Tuning complete! Now sampling."
			sims_to_do = nsims
		else:
			sims_to_do = 5000 # For acceptance rate tuning
		
		# Holds data on acceptance rates and uniform random numbers
		a_rate = np.zeros([sims_to_do,1])
		crit = np.random.rand(sims_to_do,1)

		post = multivariate_normal(np.zeros(len(initials)),cov_matrix)
		rnums = post.rvs()*scale
		for k in range(1,sims_to_do): 
			rnums = np.vstack((rnums,post.rvs()*scale))

		old_lik = -posterior(phi[0])

		for i in range(1,sims_to_do):
			phi_prop = phi[i-1] + rnums[i]

			lik_rat = exp(-posterior(phi_prop) - old_lik)

			if crit[i] < lik_rat:
				phi[i] = phi_prop
				a_rate[i] = 1
				old_lik = -posterior(phi[i])
			else:
				phi[i] = phi[i-1]
				a_rate[i] = 0

		acceptance = sum(a_rate)/len(a_rate)
			
		# Acceptance rate tuning 

		if acceptance > 0.4:
			scale *= 1.3
		elif acceptance < 0.234 and acceptance > 0.1:
			scale *= (1/1.3)
		elif acceptance <= 0.1 and acceptance > 0.05:
			scale *= 0.4
		elif acceptance <= 0.05 and acceptance > 0.01:
			scale *= 0.2
		elif acceptance <= 0.01:
			scale *= 0.05

		print "Acceptance rate of Metropolis-Hastings is" + str(sum(a_rate)/len(a_rate))

	chain = np.array([phi[i][0] for i in range(len(phi))])
	for m in range(1,len(initials)):
		chain = np.vstack((chain,[phi[i][m] for i in range(len(phi))]))

	mean_est = [np.mean(np.array([phi[i][j] for i in range(len(phi))])) for j in range(len(initials))]
	median_est = [np.median(np.array([phi[i][j] for i in range(len(phi))])) for j in range(len(initials))]
	upper_95_est = [np.percentile(np.array([phi[i][j] for i in range(len(phi))]),95) for j in range(len(initials))]
	lower_95_est = [np.percentile(np.array([phi[i][j] for i in range(len(phi))]),5) for j in range(len(initials))]

	return chain, mean_est, median_est, upper_95_est, lower_95_est