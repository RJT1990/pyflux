import numpy as np
import scipy.stats as ss
from math import exp

def vect_difference(function,phi):
	result = np.zeros(len(phi))
	for i in range(len(phi)):
		orig = function(phi)			
		phi[i] += 0.0001
		prop = function(phi)
		phi[i] -= 0.0001
		result[i] = (prop-orig)/0.0001
	return result

def hmc(data,posterior,nsims,initials,ses):

	phi = np.zeros([nsims,len(initials)])
	phi[0] = initials

	# Initialize parameters for while loop
	acceptance = 1
	finish = 0
	L = 10 # 10
	d = 0.1 # 0.1
	M = np.mean(ses)*1000 # set mass to rough estimate from MAP

	while (acceptance < 0.60 or acceptance > 0.70) or finish == 0:
		
		# If acceptance is in range, proceed to sample, else continue tuning
		if not (acceptance < 0.60 or acceptance > 0.70):
			finish = 1
			print ""
			print "Tuning complete! Now sampling."
			sims_to_do = nsims
		else:
			sims_to_do = 100 # For acceptance rate tuning
		
		# Holds data on acceptance rates, proposal random walks and uniform random numbers
		a_rate = np.zeros([sims_to_do,1])
		crit = np.random.rand(sims_to_do,1)
		p = np.random.randn(sims_to_do,len(initials))*M

		# Initialize potential and kinetic energies
		old_potential = -posterior(phi[0])
		old_kinetic = ss.norm.logpdf(0,loc=0,scale=M**2)*len(phi[0])

		for i in range(1,sims_to_do):

			phi[i] = phi[i-1]
			# Leap frog

			# First half step
			p[i] += -(d/2)*vect_difference(posterior,phi[i])
			phi[i] += d*p[i]/(M**2) # Assumption of N(0,1) kinetic energy

			for l in range(L-1):
				p[i] += -d*vect_difference(posterior,phi[i])
				phi[i] += d*p[i]/(M**2) # Assumption of N(0,1) kinetic energy

			# Final half step
			p[i] += -(d/2)*vect_difference(posterior,phi[i])

			potential = -posterior(phi[i])
			kinetic = np.sum(ss.norm.logpdf(p[i],loc=0,scale=M**2))

			lik_rat = exp(potential + kinetic - (old_potential + old_kinetic))

			if crit[i] < lik_rat:
				a_rate[i] = 1
				old_potential = potential
				old_kinetic = kinetic
			else:
				phi[i] = phi[i-1]
				a_rate[i] = 0

			print a_rate[i]

		acceptance = sum(a_rate)/len(a_rate)
			
		# Acceptance rate tuning (too crude - revise in future)
		if acceptance > 0.7:
			scale *= 1.5
		elif acceptance < 0.6:
			scale *= 0.666
			
		print "Acceptance rate of Metropolis-Hastings is" + str(sum(a_rate)/len(a_rate))

	chain = np.array([phi[i][0] for i in range(len(phi))])
	for m in range(1,len(initials)):
		chain = np.vstack((chain,[phi[i][m] for i in range(len(phi))]))

	mean_est = [np.mean(np.array([phi[i][j] for i in range(len(phi))])) for j in range(len(initials))]
	median_est = [np.median(np.array([phi[i][j] for i in range(len(phi))])) for j in range(len(initials))]
	upper_95_est = [np.percentile(np.array([phi[i][j] for i in range(len(phi))]),95) for j in range(len(initials))]
	lower_95_est = [np.percentile(np.array([phi[i][j] for i in range(len(phi))]),5) for j in range(len(initials))]

	return chain, mean_est, median_est, upper_95_est, lower_95_est