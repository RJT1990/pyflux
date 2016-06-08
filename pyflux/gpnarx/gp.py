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

class GP(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GAUSSIAN PROCESS MODELS ****
    """

    def __init__(self):

        # Initialize TSM object
        super(GP,self).__init__('GP')

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


