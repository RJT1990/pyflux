import scipy.stats as ss
from math import exp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

class q_Normal(object):

	def __init__(self,loc,scale):
		self.loc = loc
		self.scale = scale
		self.param_no = 2

	def sim(self,size): 
		return ss.norm.rvs(loc=self.loc,scale=exp(self.scale),size=size)

	def loc_score(self,x):
		return (x-self.loc)/(exp(self.scale)**2)

	def scale_score(self,x):
		# For scale = exp(x)
		return exp(-2*self.scale)*(x-self.loc)**2 - 1 

	# Indexed scale - for BBVI code
	def score(self,x,index):
		if index == 0:
			return self.loc_score(x)
		elif index == 1:
			return self.scale_score(x)

	def return_param(self,index):
		if index == 0:
			return self.loc
		elif index == 1:
			return self.scale

	def change_param(self,index,value):
		if index == 0:
			self.loc = value
		elif index == 1:
			self.scale = value

	def logpdf(self,x):
		return ss.norm.logpdf(x,loc=self.loc,scale=exp(self.scale))

	def plot_pdf(self):
		x = np.linspace(self.loc-exp(self.scale)*3.5,self.loc+exp(self.scale)*3.5,100)
		plt.plot(x,mlab.normpdf(x,self.loc,exp(self.scale)))
		plt.show()
