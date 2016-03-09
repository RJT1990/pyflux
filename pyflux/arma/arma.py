from .. import inference as ifr
from .. import output as op
import numpy as np
import scipy.stats as ss
from math import exp, sqrt, log
import math
from scipy import optimize
import matplotlib.pyplot as plt
from frequentist import *

class ARI(object):

	def __init__(self,ar,integ,data):
		self.ar = ar
		self.int = integ

		# Difference data
		for order in range(self.int):
			X = np.diff(data)
		self.data = X

		self.params = []

	def likelihood(self,beta,x,log):
		Y = np.array(x[self.ar:len(x)])
		X = np.ones(len(x)-self.ar)
		for i in range(self.ar):
			X = np.vstack((X,x[(self.ar-i-1):(len(x)-i-1)]))

		mu = np.matmul(np.transpose(X),beta[0:len(beta)-1])

		if log == True:
			return -sum(ss.norm.logpdf(mu,loc=Y,scale=exp(beta[len(beta)-1])))
		else:
			return sum(ss.norm.pdf(mu,loc=Y,scale=exp(beta[len(beta)-1])))

	def posterior(self,beta,x,log):
		Y = np.array(x[self.ar:len(x)])
		X = np.ones(len(x)-self.ar)
		for i in range(self.ar):
			X = np.vstack((X,x[(self.ar-i-1):(len(x)-i-1)]))

		mu = np.matmul(np.transpose(X),beta[0:len(beta)-1])

		if log == True:
			return -(ifr.lgaussian_prior(beta[0],0,3) + ifr.lgaussian_prior(beta[1],0,0.5) + ifr.lgaussian_prior(beta[2],0,0.5) + ifr.ligamma_prior(exp(beta[3]),0.5,0.5) + sum(ss.norm.logpdf(mu,loc=Y,scale=exp(beta[3]))))
		else:
			return ifr.gaussian_prior(beta[0],0,3)*ifr.gaussian_prior(beta[1],0,0.5)*ifr.gaussian_prior(beta[2],0,0.5)*ifr.igamma_prior(exp(beta[3]),0.5,0.5)*sum(ss.norm.pdf(mu,loc=Y,scale=exp(beta[3])))

	def fit(self,printer=True):
		# Difference dependent variable
		X = self.data

		phi = np.zeros(self.ar+2)
		p = optimize.minimize(self.likelihood,phi,args=(X,True),method='L-BFGS-B')
		self.params = p.x
		p = optimize.minimize(self.likelihood,p.x,args=(X,True),method='BFGS')
		p_std = np.diag(p.hess_inv)**0.5

		# Format parameters

		if printer == True:

			data = [
			    {'parm_name':'Constant', 'parm_value':round(p.x[0],4), 'parm_std': round(p_std[0],4),'parm_z': round(p.x[0]/float(p_std[0]),4),'parm_p': round(find_p_value(p.x[0]/float(p_std[0])),4),'ci': "(" + str(round(p.x[0] - p_std[0]*1.96,4)) + " | " + str(round(p.x[0] + p_std[0]*1.96,4)) + ")"}
			]

			for k in range(self.ar):
				data.append({'parm_name':'AR(' + str(k+1) + ')', 'parm_value':round(p.x[k+1],4), 'parm_std': round(p_std[k+1],4), 
					'parm_z': round(p.x[k+1]/float(p_std[k+1]),4), 'parm_p': round(find_p_value(p.x[k+1]/float(p_std[k+1])),4),'ci': "(" + str(round(p.x[k+1] - p_std[k+1]*1.96,4)) + " | " + str(round(p.x[k+1] + p_std[k+1]*1.96,4)) + ")"})

			fmt = [
			    ('Parameter',       'parm_name',   20),
			    ('Estimate',          'parm_value',       10),
			    ('Standard Error', 'parm_std', 15),
			    ('z',          'parm_z',       10),
			    ('P>|z|',          'parm_p',       10),
			    ('95% Confidence Interval',          'ci',       25)
			]

			print "AR(" + str(self.ar) + ") regression"
			print "=================="
			print "Method: MLE"
			print "Number of observations: " + str(len(X)-self.ar)
			print "Log Likelihood: " + str(round(-self.likelihood(p.x,X,True),4))
			print "AIC: " + str(round(2*len(p.x)+2*self.likelihood(p.x,X,True),4))
			print "BIC: " + str(round(2*self.likelihood(p.x,X,True) + len(p.x)*log(len(X)-self.ar),4))
			print ""
			print( op.TablePrinter(fmt, ul='=')(data) )

	def mh_fit(self,alpha=0.234,scale=(2.38/sqrt(4)),nsims=100000):
		X = self.data
		self.fit(printer=False)
		phi = np.zeros([nsims,self.ar+2])
		a_rate = np.zeros([nsims,1])
		phi[0] = self.params
		rnums = np.random.randn(nsims,self.ar+2)*scale
		crit = np.random.rand(nsims,1)
		for i in range(1,nsims):

			phi_prop = phi[i-1] + rnums[i]

			prior_old = ifr.gaussian_prior(phi[i][0],0,3)*ifr.gaussian_prior(phi[i][1],0,0.5)*ifr.gaussian_prior(phi[i][2],0,0.5)*ifr.igamma_prior(exp(phi[i][3]),0.5,0.5)
			prior_new = ifr.gaussian_prior(phi_prop[0],0,3)*ifr.gaussian_prior(phi_prop[1],0,0.5)*ifr.gaussian_prior(phi_prop[2],0,0.5)*ifr.igamma_prior(exp(phi_prop[3]),0.5,0.5)

			lik_rat = (self.likelihood(phi_prop,X,False)*prior_new)/(self.likelihood(phi[i-1],X,False)*prior_old)

			if crit[i] < min(lik_rat,1):
				phi[i] = phi_prop
				a_rate[i] = 1
			else:
				phi[i] = phi[i-1]
				a_rate[i] = 0


		print "Acceptance rate of Metropolis-Hastings", sum(a_rate)/len(a_rate)

		chain = np.array([phi[i][0] for i in range(len(phi))])
		for m in range(1,4):
			chain = np.vstack((chain,[phi[i][m] for i in range(len(phi))]))

		plt.figure(1)
		start_plot = 330
		nparm = 3
		for j in range(nparm):
			for k in range(3):
				it_num = start_plot + j*3 + k + 1
				a = plt.subplot(it_num)
				if it_num-start_plot in [1,4,7]:
					a.set_title('Parameter' + str(j) + 'Histogram')
					normed_value = 2
					hist, bins = np.histogram([phi[i][j] for i in range(len(phi))], density=True)
					widths = np.diff(bins)
					hist *= normed_value
					plt.bar(bins[:-1], hist, widths)	
				elif it_num-start_plot in [2,5,8]:
					a.set_title('Parameter' + str(j) + 'Chain')
					plt.plot(chain[j])
				elif it_num-start_plot in [3,6,9]:
					a.set_title('Cumulative Average')
					plt.plot(np.cumsum(np.array([phi[i][j] for i in range(len(phi))]))/np.array(range(1,len(phi)+1)))

		plt.show()				


	def predict(self,T=5):
		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			predictions = self.data

			for t in range(T):
				new_value = self.params[0]
				for k in range(self.ar):
					new_value += self.params[k+1]*predictions[len(predictions)-1-k]
				predictions = np.append(predictions,[new_value])
			return predictions





