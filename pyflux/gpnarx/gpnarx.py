import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns

from .. import arma
from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .kernels import *

class GPNARX(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GAUSSIAN PROCESS NONLINEAR AUTOREGRESSIVE (GP-NARX) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    ar : int
        Field to specify how many AR terms the model will have.

    kernel_type : str
        One of SE (SquaredExponential), OU (Ornstein-Uhlenbeck), RQ
        (RationalQuadratic), Periodic, ARD. Defines kernel choice for GP-NARX.

    integ : int (default : 0)
        Specifies how many time to difference the time series.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.
    """

    def __init__(self,data,ar,kernel_type='SE',integ=0,target=None):

        # Initialize TSM object
        super(GPNARX,self).__init__('GPNARX')

        # Parameters
        self.ar = ar
        self.integ = integ

        if kernel_type == 'ARD':
            self.param_no = 2 + self.ar
        elif kernel_type == 'RQ':
            self.param_no = 3 + 1
        else:
            self.param_no = 3

        self.max_lag = self.ar
        self.model_name = 'GPNARX(' + str(self.ar) + ')'
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"     
        self.kernel_type = kernel_type
        self.multivariate_model = False

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data_original = self.data.copy()

        # Difference data
        for order in range(self.integ):
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name
        self.index = self.index[self.integ:len(self.index)]

        # Apply normalization
        self.data_full = self.data.copy()       
        self.data = np.array(self.data_full[self.max_lag:self.data_full.shape[0]]) # adjust for lags
        self._norm_mean = np.mean(self.data)
        self._norm_std = np.std(self.data)  
        self.data = (self.data - self._norm_mean) / self._norm_std
        self.data_full = (self.data_full - self._norm_mean) / self._norm_std

        # Define parameters
        self._create_parameters()

    def _alpha(self,L):
        """ Covariance-derived term to construct expectations. See Rasmussen & Williams.

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        The alpha matrix/vector
        """     

        return np.linalg.solve(np.transpose(L),np.linalg.solve(L,np.transpose(self.data)))

    def _construct_predict(self,beta,h):    
        """ Creates h-step ahead forecasts for the Gaussian process

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        h: int
            How many steps ahead to forecast

        Returns
        ----------
        - predictions
        - variance of predictions
        """             

        # Refactor this entire code
        self._start_params(beta)
        Xstart = self.X().copy()
        Xstart = [i for i in Xstart]
        predictions = np.zeros(h)
        variances = np.zeros(h)

        for step in range(0,h):
            Xstar = []

            for lag in range(0,self.max_lag):
                if lag == 0:
                    if step == 0:
                        Xstar.append([self.data[-1]])
                        Xstart[0] = np.append(Xstart[0],self.data[-1])
                    else:
                        Xstar.append([predictions[step-1]])
                        Xstart[0] = np.append(Xstart[0],predictions[step-1])
                else:
                    Xstar.append([Xstart[lag-1][-2]])
                    Xstart[lag] = np.append(Xstart[lag],Xstart[lag-1][-2])

            Kstar = self.kernel.Kstar(np.transpose(np.array(Xstar)))

            L = self._L(beta)
            
            predictions[step] = np.dot(np.transpose(Kstar),self._alpha(L))
            variances[step] = self.kernel.Kstarstar(np.transpose(np.array(Xstar))) 
            - np.dot(np.dot(np.transpose(self.kernel.Kstar(np.transpose(np.array(Xstar)))),
                np.linalg.pinv(self.kernel.K() + np.identity(self.kernel.K().shape[0])*self.parameters.parameter_list[0].prior.transform(beta[0]))),
                self.kernel.Kstar(np.transpose(np.array(Xstar)))) + self.parameters.parameter_list[0].prior.transform(beta[0])  

        return predictions, variances, predictions - 1.98*np.power(variances,0.5), predictions + 1.98*np.power(variances,0.5)

    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        self.parameters.add_parameter('Noise Sigma^2',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        if self.kernel_type == 'ARD':
            self.kernel = ARD(self.X(),np.ones(self.ar),1)

            for lag in range(self.ar):
                self.parameters.add_parameter('l lag-' + str(lag+1),ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        else:
            self.parameters.add_parameter('l',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

            if self.kernel_type == 'SE':
                self.kernel = SquaredExponential(self.X(),1,1)
            elif self.kernel_type == 'OU':
                self.kernel = OrnsteinUhlenbeck(self.X(),1,1)
            elif self.kernel_type == 'Periodic':
                self.kernel = Periodic(self.X(),1,1)
            elif self.kernel_type == 'RQ':
                self.parameters.add_parameter('alpha',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
                self.kernel = RationalQuadratic(self.X(),1,1,1)

        self.parameters.add_parameter('tau',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        arma_start = arma.ARIMA(self.data,ar=self.ar,ma=0,integ=self.integ)
        x = arma_start.fit()
        arma_starting_values = arma_start.parameters.get_parameter_values()

        for parameter in range(len(self.parameters.parameter_list)):
            self.parameters.parameter_list[parameter].start = -1.0

        self.parameters.parameter_list[0].start = np.log(np.exp(np.power(arma_starting_values[-1],2)))

    def _L(self,beta):
        """ Creates cholesky decomposition of covariance matrix

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        The cholesky decomposition (L) of K
        """ 

        return np.linalg.cholesky(self.kernel.K()) + np.identity(self.kernel.K().shape[0])*self.parameters.parameter_list[0].prior.transform(beta[0])

    def _start_params(self,beta):
        """ Transforms parameters for use in kernels

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        None (changes data in self.kernel)
        """

        if self.kernel_type == 'ARD':
            for lag in range(0,self.ar):
                self.kernel.l[lag] = self.parameters.parameter_list[1+lag].prior.transform(beta[1+lag])
            self.kernel.tau = self.parameters.parameter_list[-1].prior.transform(beta[-1])  
        elif self.kernel_type == 'RQ':
            self.kernel.l = self.parameters.parameter_list[1].prior.transform(beta[1])
            self.kernel.a = self.parameters.parameter_list[2].prior.transform(beta[2])          
            self.kernel.tau = self.parameters.parameter_list[3].prior.transform(beta[3])    
        else:
            self.kernel.l = self.parameters.parameter_list[1].prior.transform(beta[1])
            self.kernel.tau = self.parameters.parameter_list[2].prior.transform(beta[2])            

    def X(self):
        """ Creates design matrix of variables to use in GP regression

        Returns
        ----------
        The design matrix
        """     

        for i in range(0,self.ar):
            datapoint = self.data_full[(self.max_lag-i-1):-i-1]         
            if i==0:
                X = datapoint
            else:
                X = np.vstack((X,datapoint))
        return X

    def expected_values(self,beta):
        """ Expected values of the function given the covariance matrix and hyperparameters

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        The expected values of the function
        """     

        self._start_params(beta)
        L = self._L(beta)
        return np.dot(np.transpose(self.kernel.K()),self._alpha(L))

    def variance_values(self,beta):
        """ Covariance matrix for the estimated function

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        Covariance matrix for the estimated function 
        """     

        self._start_params(beta)

        return self.kernel.K() - np.dot(np.dot(np.transpose(self.kernel.K()),np.linalg.pinv(self.kernel.K() + np.identity(self.kernel.K().shape[0])*self.parameters.parameter_list[0].prior.transform(beta[0]))),self.kernel.K()) + self.parameters.parameter_list[0].prior.transform(beta[0])

    def neg_loglik(self,beta):
        """ Creates the negative log marginal likelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        The negative log marginal logliklihood of the model
        """             
        self._start_params(beta)
        L = self._L(beta)
        return -(-0.5*(np.dot(np.transpose(self.data),self._alpha(L))) - np.trace(L) - (self.data.shape[0]/2)*np.log(2*np.pi))

    def plot_fit(self,intervals=True,**kwargs):
        """ Plots the fit of the Gaussian process model to the data

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        intervals : Boolean
            Whether to plot uncertainty intervals or not

        Returns
        ----------
        None (plots the fit of the function)
        """

        figsize = kwargs.get('figsize',(10,7))

        date_index = self.index[self.max_lag:]
        expectation = self.expected_values(self.parameters.get_parameter_values())
        posterior = multivariate_normal(expectation,self.variance_values(self.parameters.get_parameter_values()),allow_singular=True)
        simulations = 500
        sim_vector = np.zeros([simulations,len(expectation)])

        for i in range(simulations):
            sim_vector[i] = posterior.rvs()

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append([(np.percentile(i,pre)*self._norm_std + self._norm_mean) for i in sim_vector.transpose()] 
                - (expectation*self._norm_std + self._norm_mean))

        plt.figure(figsize=figsize) 

        plt.subplot(2, 2, 1)
        plt.title(self.data_name + " Raw")  
        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k')

        plt.subplot(2, 2, 2)
        plt.title(self.data_name + " Raw and Expected") 
        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k',alpha=0.2)
        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.subplot(2, 2, 3)
        plt.title(self.data_name + " Raw and Expected (with intervals)")    

        if intervals == True:
            alpha =[0.15*i/float(100) for i in range(50,12,-2)]
            for count, pre in enumerate(error_bars):
                plt.fill_between(date_index, (expectation*self._norm_std + self._norm_mean)-pre, 
                    (expectation*self._norm_std + self._norm_mean)+pre,alpha=alpha[count])      

        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k',alpha=0.2)
        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.subplot(2, 2, 4)

        plt.title("Expected " + self.data_name + " (with intervals)")   

        if intervals == True:
            alpha =[0.15*i/float(100) for i in range(50,12,-2)]
            for count, pre in enumerate(error_bars):
                plt.fill_between(date_index, (expectation*self._norm_std + self._norm_mean)-pre, 
                    (expectation*self._norm_std + self._norm_mean)+pre,alpha=alpha[count])      

        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.show()

    def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):

        """ Plots forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show 95% prediction intervals for the forecast?

        Returns
        ----------
        - Plot of the forecast
        - Error bars, forecasted_values, plot_values, plot_index
        """     

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:

            predictions, variance, lower, upper = self._construct_predict(self.parameters.get_parameter_values(),h) 
            full_predictions = np.append(self.data,predictions)
            full_lower = np.append(self.data,lower)
            full_upper = np.append(self.data,upper)
            date_index = self.shift_dates(h)

            # Plot values (how far to look back)
            plot_values = full_predictions[-h-past_values:]*self._norm_std + self._norm_mean
            plot_index = date_index[-h-past_values:]

            # Lower and upper intervals
            lower = np.append(full_predictions[-h-1],lower)
            upper = np.append(full_predictions[-h-1],upper)

            plt.figure(figsize=figsize)
            if intervals == True:
                plt.fill_between(date_index[-h-1:], 
                    lower*self._norm_std + self._norm_mean, 
                    upper*self._norm_std + self._norm_mean,
                    alpha=0.2)          
            
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
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
            x = GPNARX(ar=self.ar,kernel_type=self.kernel_type,integ=self.integ,data=self.data_original[:-h+t])
            if t == 0:
                x.fit(printer=False)
                predictions = x.predict(1)  
            else:
                x.fit(printer=False)
                predictions = pd.concat([predictions,x.predict(1)])

        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

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
        date_index = self.index[-h:]
        predictions = self.predict_is(h)
        data = self.data[-h:]

        plt.plot(date_index,data*self._norm_std + self._norm_mean,label='Data')
        plt.plot(date_index,predictions,label='Predictions',c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def predict(self,h=5):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:

            predictions, _, _, _ = self._construct_predict(self.parameters.get_parameter_values(),h)    
            predictions = predictions*self._norm_std + self._norm_mean  
            date_index = self.shift_dates(h)
            result = pd.DataFrame(predictions)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result


