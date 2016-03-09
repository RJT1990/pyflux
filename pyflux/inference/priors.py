from math import exp, log

def gaussian_prior(mu,mu0,sig0):
	return (1/float(sig0))*exp(-0.5*((mu-mu0)**2)/float(sig0**2))

def igamma_prior(z,alpha,beta):
	return (z**(-alpha-1))*exp(-(beta/float(z)))

def lgaussian_prior(mu,mu0,sig0):
	return -log(float(sig0)) - 0.5*((mu-mu0)**2)/float(sig0**2)

def ligamma_prior(z,alpha,beta):
	return (-alpha-1)*log(z) -(beta/float(z))