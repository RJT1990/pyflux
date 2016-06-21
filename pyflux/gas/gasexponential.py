import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

from ..parameter import Parameter, Parameters
from .. import inference as ifr
from .. import tsm as tsm
from .. import distributions as dst
from .. import data_check as dc

from .scores import *
from .gas import *

class GASExponential(GAS):
    """ Inherits GAS methods from GAS class (and time series methods from TSM class).

    **** EXPONENTIAL GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the univariate time series data that will be used.

    ar : int
        Field to specify how many AR lags the model will have.

    sc : int
        Field to specify how many score lags terms the model will have.

    integ : int (default : 0)
        Specifies how many time to difference the time series.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.

    gradient_only : Boolean (default: True)
        If true, will only use gradient rather than second-order terms
        to construct the modified score.
    """

    def __init__(self,data,ar,sc,integ=0,target=None,gradient_only=False):

        # Initialize TSM object     
        super(GASExponential,self).__init__(data=data,ar=ar,sc=sc,integ=integ,
            target=target,gradient_only=gradient_only)

        self.model_name = "EXPONENTIAL GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") REGRESSION"
        self.dist = 'Exponential'
        self.link = np.exp
        self.scale = False
        self.shape = False
        self.parameters.parameter_list[0].start = np.log(1/np.mean(self.data))

        if gradient_only is False:
            self.score_function = self.adj_score_function
        else:
            self.score_function = self.default_score_function

    def _mean_prediction(self,theta,Y,scores,h,t_params):
        """ Creates a h-step ahead mean prediction

        Parameters
        ----------
        theta : np.array
            The past predicted values

        Y : np.array
            The past data

        scores : np.array
            The past scores

        h : int
            How many steps ahead for the prediction

        t_params : np.array
            A vector of (transformed) parameters

        Returns
        ----------
        Y_exp : np.array
            Vector of past values and predictions 
        """     

        Y_exp = Y.copy()
        theta_exp = theta.copy()
        scores_exp = scores.copy()

        #(TODO: vectorize the inner construction here)      
        for t in range(0,h):
            new_value = t_params[0]

            if self.ar != 0:
                for j in range(1,self.ar+1):
                    new_value += t_params[j]*theta_exp[-j]

            if self.sc != 0:
                for k in range(1,self.sc+1):
                    new_value += t_params[k+self.ar]*scores_exp[-k]

            Y_exp = np.append(Y_exp,[1/self.link(new_value)])
            theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
            scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

        return Y_exp

    def adj_score_function(self,y,mean,scale,shape,skewness):
        return ExponentialScore.log_lam_adj_score(y, mean)

    def draw_variable(self,loc,scale,shape,skewness,nsims):
        return np.random.exponential(1/loc, nsims)

    def neg_loglik(self,beta):
        theta, Y, scores = self._model(beta)
        return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))

    def default_score_function(self,y,mean,scale,shape,skewness):
        return ExponentialScore.log_lam_score(y, mean)

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
            x = GASExponential(ar=self.ar,sc=self.sc,integ=self.integ,data=self.data_original[:-h+t])
            x.fit(printer=False)
            
            if t == 0:
                predictions = x.predict(1)
            else:
                predictions = pd.concat([predictions,x.predict(1)])
        
        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

    def plot_fit(self,intervals=False,**kwargs):
        """ Plots the fit of the model

        Returns
        ----------
        None (plots data and the fit)
        """

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:
            date_index = self.index[max(self.ar,self.sc):]
            mu, Y, scores = self._model(self.parameters.get_parameter_values())

            if intervals == True:
                sim_vector = self.link([self._bootstrap_scores(self.parameters.get_parameter_values()) for i in range(1000)]).T
                error_bars = []
                error_bars.append(1/np.array([np.percentile(i,5) for i in sim_vector]))
                error_bars.append(1/np.array([np.percentile(i,95) for i in sim_vector]))

            plt.figure(figsize=figsize)
            plt.subplot(2,1,1)
            plt.title("Model fit for " + self.data_name)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, error_bars[0], error_bars[1], alpha=0.15,label='95% Confidence Interval')  

            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,1/self.link(mu),label='GAS Filter',c='black')
            plt.legend(loc=2)   

            plt.subplot(2,1,2)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, error_bars[0], error_bars[1], alpha=0.15,label='95% Confidence Interval')  

            plt.plot(date_index,1/self.link(mu),label='GAS Filter',c='black')
            plt.title("Filtered values for " + self.data_name)
            plt.legend(loc=2)   

            plt.show()              
    
    def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show prediction intervals for the forecast?

        Returns
        ----------
        - Plot of the forecast
        """     

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:

            # Retrieve data, dates and (transformed) parameters
            theta, Y, scores = self._model(self.parameters.get_parameter_values())          
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(theta,Y,scores,h,t_params)
            sim_values = self._sim_prediction(theta,Y,scores,h,t_params,15000)
            error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,
                        alpha=alpha[count])         
            
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
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

            theta, Y, scores = self._model(self.parameters.get_parameter_values())          
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            mean_values = self._mean_prediction(theta,Y,scores,h,t_params)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(1/forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result