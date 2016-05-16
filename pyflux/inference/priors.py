from math import exp, log, tanh
import numpy as np

def ilogit(x):
	return 1/(1+np.exp(-x))

def logit(x):
	return np.log(x) - np.log(1 - x)

def transform_define(transform):
	if transform == 'tanh':
		return np.tanh
	elif transform == 'exp':
		return np.exp
	elif transform == 'logit':
		return ilogit
	elif transform is None:
		return np.array
	else:
		return None

def itransform_define(transform):
	if transform == 'tanh':
		return np.arctanh
	elif transform == 'exp':
		return np.log
	elif transform == 'logit':
		return logit
	elif transform is None:
		return np.array
	else:
		return None

class Normal(object):

	def __init__(self,mu0,sigma0,transform=None):
		self.mu0 = mu0
		self.sigma0 = sigma0
		self.transform_name = transform		
		self.transform = transform_define(transform)
		self.itransform = itransform_define(transform)

	def logpdf(self,mu):
		if self.transform is not None:
			mu = self.transform(mu)		
		return -log(float(self.sigma0)) - (0.5*(mu-self.mu0)**2)/float(self.sigma0**2)

	def pdf(self,mu):
		if self.transform is not None:
			mu = self.transform(mu)				
		return (1/float(self.sigma0))*exp(-(0.5*(mu-self.mu0)**2)/float(self.sigma0**2))

class Uniform(object):

	def __init__(self,transform=None):
		self.transform_name = transform		
		self.transform = transform_define(transform)
		self.itransform = itransform_define(transform)

	def logpdf(self,mu):
		return 0.0

class InverseGamma(object):

	def __init__(self,alpha,beta,transform=np.exp):
		self.alpha = alpha
		self.beta = beta
		self.transform_name = transform
		self.transform = transform_define(transform)
		self.itransform = itransform_define(transform)

	def logpdf(self,x):
		if self.transform is not None:
			x = self.transform(x)		
		return (-self.alpha-1)*log(x) - (self.beta/float(x))

	def pdf(self,x):
		if self.transform is not None:
			x = self.transform(x)				
		return (x**(-self.alpha-1))*exp(-(self.beta/float(x)))


