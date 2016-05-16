from math import exp, sqrt, log, tanh
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from patsy import dmatrices, dmatrix, demo_data

from .. import inference as ifr
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import distributions as dst
from .. import data_check as dc

from .scores import *

class GASReg(tsm.TSM):
	""" Inherits time series methods from TSM class.

	**** GENERALIZED AUTOREGRESSIVE SCORE (GAS) REGRESSION MODELS ****

	Parameters
	----------

	formula : string
		patsy string describing the regression

	data : pd.DataFrame or np.array
		Field to specify the data that will be used
	"""

	def __init__(self,formula,data):

		# Initialize TSM object		
		super(GASReg,self).__init__('GASReg')

		# Parameters
		self.max_lag = 0
		self._hess_type = 'numerical'
		self._param_hide = 0 # Whether to cutoff variance parameters from results
		self.supported_methods = ["MLE","MAP","Laplace","M-H","BBVI"]
		self.default_method = "MLE"

		# Format the data
		self.is_pandas = True # This is compulsory for this model type
		self.data_original = data
		self.formula = formula
		self.y, self.X = dmatrices(formula, data)
		self.param_no = self.X.shape[1]
		self.y_name = self.y.design_info.describe()
		self.X_names = self.X.design_info.describe().split(" + ")
		self.y = np.array([self.y]).ravel()
		self.data = self.y
		self.X = np.array([self.X])[0]
		self.index = data.index

	def _model(self,beta):
		""" Creates the structure of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		theta : np.array
			Contains the predicted values for the time series

		Y : np.array
			Contains the length-adjusted time series (accounting for lags)

		scores : np.array
			Contains the scores for the time series
		"""

		Y = self.data
		scores = np.zeros((self.X.shape[1],Y.shape[0]+1))
		parm = np.array([self._param_desc[k]['prior'].transform(beta[k]) for k in range(beta.shape[0])])
		coefficients = np.zeros((self.X.shape[1],Y.shape[0]+1))
		coefficients[:,0] = self.initial_params
		theta = np.zeros(Y.shape[0]+1)

		# Check if model has scale parameter
		if self.scale is True:
			if self.dist == 't':
				model_v = parm[parm.shape[0]-2]	
			else:
				model_v = 0		
			model_scale = parm[parm.shape[0]-1]
		else:
			model_scale = 0
			model_v = 0

		# Loop over time series
		for t in range(0,Y.shape[0]):
			theta[t] = np.dot(self.X[t],coefficients[:,t])
			scores[:,t] = self.score_function(self.X[t],Y[t],theta[t],model_scale,model_v)
			coefficients[:,t+1] = coefficients[:,t] + parm[0:self.X.shape[1]]*scores[:,t] 
		return theta[0:theta.shape[0]-1], Y, scores[0:theta.shape[0]-1], coefficients

	@classmethod
	def Exponential(cls,formula,data):
		""" Creates Exponential-distributed GASReg model

		Parameters
		----------
		formula : string
			patsy string describing the regression

		data : np.array
			Contains the time series

		Returns
		----------
		- GASReg.Exponential object
		"""		

		x = GASReg(formula=formula,data=data)

		for parm in range(x.param_no):
			x._param_desc.append({'name' : 'Scale ' + x.X_names[parm],'index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		def Exponential_score_function(X,y,theta,scale,v):
			return X*(1.0 - theta*y)

		x.score_function = Exponential_score_function
		x.model_name = "Exponential-distributed GAS Regression"
		x.link = np.exp
		x.starting_params = np.zeros(x.param_no)
		x.starting_params[0:x.param_no] = -9
		x.initial_params = np.zeros(x.param_no)
		x.scale = False
		x.dist = 'Exponential'
		return x

	@classmethod
	def Laplace(cls,formula,data):
		""" Creates Laplace-distributed GASReg model

		Parameters
		----------
		formula : string
			patsy string describing the regression

		data : np.array
			Contains the time series

		Returns
		----------
		- GASReg.Laplace object
		"""		

		x = GASReg(formula=formula,data=data)

		for parm in range(x.param_no):
			x._param_desc.append({'name' : 'Scale ' + x.X_names[parm],'index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		x._param_desc.append({'name' : 'Laplace-scale','index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})			

		def Laplace_score_function(X,y,theta,scale,v):
			return X*(y-theta)/(scale*np.abs(y-theta))

		x.score_function = Normal_score_function
		x.model_name = "Laplace-distributed GAS Regression"
		x.link = np.array
		x.initial_params = np.zeros(x.param_no)
		x.param_no += 1
		x.starting_params = np.zeros(x.param_no)
		x.starting_params[0:x.param_no] = -9
		x.starting_params[len(x._param_desc)-1] = 0.0
		x.scale = True
		x.dist = 'Laplace'
		return x

	@classmethod
	def Normal(cls,formula,data):
		""" Creates Normal-distributed GASReg model

		Parameters
		----------
		formula : string
			patsy string describing the regression

		data : np.array
			Contains the time series

		Returns
		----------
		- GASReg.Normal object
		"""		

		x = GASReg(formula=formula,data=data)

		for parm in range(x.param_no):
			x._param_desc.append({'name' : 'Scale ' + x.X_names[parm],'index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		x._param_desc.append({'name' : 'Normal-scale','index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})			

		def Normal_score_function(X,y,theta,scale,v):
			return X*(y-theta)

		x.score_function = Normal_score_function
		x.model_name = "Normal-distributed GAS Regression"
		x.link = np.array
		x.initial_params = np.zeros(x.param_no)
		x.param_no += 1
		x.starting_params = np.zeros(x.param_no)
		x.starting_params[0:x.param_no] = -9
		x.starting_params[len(x._param_desc)-1] = 0.0
		x.scale = True
		x.dist = 'Normal'
		x.dist_function = GASReg.Normal
		return x

	@classmethod
	def Poisson(cls,formula,data):
		""" Creates Poisson-distributed GASReg model

		Parameters
		----------
		formula : string
			patsy string describing the regression

		data : np.array
			Contains the time series

		Returns
		----------
		- GASReg.Poisson object
		"""		

		x = GASReg(formula=formula,data=data)

		for parm in range(x.param_no):
			x._param_desc.append({'name' : 'Scale ' + x.X_names[parm],'index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		def Poisson_score_function(X,y,theta,scale,v):
			return X*(y-np.exp(theta))

		x.score_function = Poisson_score_function
		x.model_name = "Poisson-distributed GAS Regression"
		x.link = np.exp
		x.initial_params = np.zeros(x.param_no)
		x.starting_params = np.zeros(x.param_no)
		x.starting_params[0:x.param_no] = -9
		x.scale = False
		x.dist = 'Poisson'
		return x

	@classmethod
	def t(cls,formula,data):
		""" Creates t-distributed GASReg model

		Parameters
		----------
		formula : string
			patsy string describing the regression

		data : np.array
			Contains the time series

		Returns
		----------
		- GASReg.t object
		"""		

		x = GASReg(formula=formula,data=data)

		for parm in range(x.param_no):
			x._param_desc.append({'name' : 'Scale ' + x.X_names[parm],'index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})

		x._param_desc.append({'name' : 'v','index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})			
		x._param_desc.append({'name' : 't-scale','index': len(x._param_desc), 'prior': ifr.Uniform(transform='exp'), 'q': dst.q_Normal(0,3)})			

		def t_score_function(X,y,theta,scale,v):
			return ((v+1)/v)*((y-theta)*X)/(np.power(scale,2)+np.power((y-theta)/v,2))

		x.score_function = t_score_function
		x.model_name = "t-distributed GAS Regression"
		x.link = np.array
		x.initial_params = np.zeros(x.param_no)
		x.param_no += 2
		x.starting_params = np.zeros(x.param_no)	
		x.starting_params[0:x.param_no] = -9
		x.starting_params[len(x._param_desc)-2] = 2.0
		x.starting_params[len(x._param_desc)-1] = 0.0
		x.scale = True
		x.dist = 't'
		return x

	def likelihood(self,beta):
		""" Creates the negative log-likelihood of the model

		Parameters
		----------
		beta : np.array
			Contains untransformed starting values for parameters

		Returns
		----------
		The negative logliklihood of the model
		"""		

		theta, Y, scores,_ = self._model(beta)

		if self.dist == "Laplace":
			return -np.sum(ss.laplace.logpdf(Y,loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))
		elif self.dist == "Normal":
			return -np.sum(ss.norm.logpdf(Y,loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))	
		elif self.dist == "Poisson":
			return -np.sum(ss.poisson.logpmf(Y,self.link(theta)))
		elif self.dist == "Exponential":
			return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))
		elif self.dist == "t":
			return -np.sum(ss.t.logpdf(x=Y,df=self._param_desc[beta.shape[0]-2]['prior'].transform(beta[beta.shape[0]-2]),loc=theta,scale=self._param_desc[beta.shape[0]-1]['prior'].transform(beta[beta.shape[0]-1])))

	def plot_fit(self,**kwargs):
		""" Plots the fit of the model

		Returns
		----------
		None (plots data and the fit)
		"""

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			date_index = self.index.copy()
			mu, Y, scores, coefficients = self._model(self.params)

			plt.figure(figsize=figsize)	
			
			plt.subplot(len(self.X_names)+1, 1, 1)
			plt.title(self.y_name + " Filtered")
			plt.plot(date_index,Y,label='Data')
			plt.plot(date_index,self.link(mu),label='Filter',c='black')
			plt.legend(loc=2)

			for coef in range(0,len(self.X_names)):
				plt.subplot(len(self.X_names)+1, 1, 2+coef)
				plt.title("Beta " + self.X_names[coef])	
				plt.plot(date_index,coefficients[coef,0:coefficients.shape[1]-1],label='Coefficient')
				plt.legend(loc=2)				

			plt.show()			
	
	def plot_predict(self,h=5,past_values=20,intervals=True,oos_data=None,**kwargs):
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		past_values : int (default : 20)
			How many past observations to show on the forecast graph?

		intervals : Boolean
			Would you like to show prediction intervals for the forecast?

		oos_data : pd.DataFrame
			Data for the variables to be used out of sample (ys can be NaNs)

		Returns
		----------
		- Plot of the forecast
		"""		

		figsize = kwargs.get('figsize',(10,7))

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Sort/manipulate the out-of-sample data
			_, X_oos = dmatrices(self.formula, oos_data)
			X_oos = np.array([X_oos])[0]
			X_pred = X_oos[0:h]
			date_index = self.shift_dates(h)
			_, _, _, coefficients = self._model(self.params)	
			coefficients_star = coefficients.T[coefficients.T.shape[0]-1]
			theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]
			t_params = np.array([self._param_desc[k]['prior'].transform(self.params[k]) for k in range(self.params.shape[0])])

			# Measurement prediction intervals
			if self.dist == "Normal":
				rnd_value = np.random.normal(theta_pred, t_params[t_params.shape[0]-1], [15000,theta_pred.shape[0]])
			elif self.dist == "Laplace":
				rnd_value = np.random.laplace(theta_pred, t_params[t_params.shape[0]-1],[15000,theta_pred.shape[0]])
			elif self.dist == "Poisson":
				rnd_value = np.random.poisson(self.link(theta_pred), [15000,theta_pred.shape[0]])
			elif self.dist == "Exponential":
				rnd_value = np.random.exponential(1/self.link(theta_pred), [15000,theta_pred.shape[0]])
			elif self.dist == 't':
				rnd_value = theta_pred + t_params[t_params.shape[0]-1]*np.random.standard_t(t_params[t_params.shape[0]-2],[15000,theta_pred.shape[0]])

			error_bars = []
			for pre in range(5,100,5):
				error_bars.append(np.insert([np.percentile(i,pre) for i in rnd_value.T] - self.link(theta_pred),0,0))

			plot_values = np.append(self.y,self.link(theta_pred))
			plot_values = plot_values[len(plot_values)-h-past_values:len(plot_values)]
			forecasted_values = np.append(self.y[self.y.shape[0]-1],self.link(theta_pred))
			plot_index = date_index[len(date_index)-h-past_values:len(date_index)]

			plt.figure(figsize=figsize)
			if intervals == True:
				alpha =[0.15*i/float(100) for i in range(50,12,-2)]
				for count, pre in enumerate(error_bars):
					plt.fill_between(date_index[len(date_index)-h-1:len(date_index)], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])			
			
			plt.plot(plot_index,plot_values)
			plt.title("Forecast for " + self.y_name)
			plt.xlabel("Time")
			plt.ylabel(self.y_name)
			plt.show()

	def predict_is(self,h=5):
		""" Makes dynamic in-sample predictions with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps would you like to forecast?

		Returns
		----------
		- pd.DataFrame with predicted values
		"""		

		predictions = []

		for t in range(0,h):
			data1 = self.data_original.iloc[0:self.data_original.shape[0]-h+t,:]
			data2 = self.data_original.iloc[self.data_original.shape[0]-h+t:self.data_original.shape[0],:]
			x = self.dist_function(formula=self.formula,data=self.data_original[0:(self.data_original.shape[0]-h+t)])
			x.fit(printer=False)
			if t == 0:
				predictions = x.predict(1,oos_data=data2)
			else:
				predictions = pd.concat([predictions,x.predict(h=1,oos_data=data2)])
		
		predictions.rename(columns={0:self.y_name}, inplace=True)
		predictions.index = self.index[(len(self.index)-h):len(self.index)]

		return predictions

	def plot_predict_is(self,h=5,**kwargs):
		""" Plots forecasts with the estimated model against data
			(Simulated prediction with data)

		Parameters
		----------
		h : int (default : 5)
			How many steps to forecast

		Returns
		----------
		- Plot of the forecast against data 
		"""		

		figsize = kwargs.get('figsize',(10,7))

		plt.figure(figsize=figsize)
		predictions = self.predict_is(h)
		data = self.data[(len(self.data)-h):len(self.data)]

		plt.plot(predictions.index,data,label='Data')
		plt.plot(predictions.index,predictions,label='Predictions',c='black')
		plt.title(self.y_name)
		plt.legend(loc=2)	
		plt.show()			

	def predict(self,h=5,oos_data=None):
		""" Makes forecast with the estimated model

		Parameters
		----------
		h : int (default : 5)
			How many steps ahead would you like to forecast?

		oos_data : pd.DataFrame
			Data for the variables to be used out of sample (ys can be NaNs)

		Returns
		----------
		- pd.DataFrame with predicted values
		"""		

		if len(self.params) == 0:
			raise Exception("No parameters estimated!")
		else:

			# Sort/manipulate the out-of-sample data
			_, X_oos = dmatrices(self.formula, oos_data)
			X_oos = np.array([X_oos])[0]
			X_pred = X_oos[0:h]
			date_index = self.shift_dates(h)
			_, _, _, coefficients = self._model(self.params)	
			coefficients_star = coefficients.T[coefficients.T.shape[0]-1]
			theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]

			result = pd.DataFrame(self.link(theta_pred))
			result.rename(columns={0:self.y_name}, inplace=True)
			result.index = date_index[len(date_index)-h:len(date_index)]

			return result