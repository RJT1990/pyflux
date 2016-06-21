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

class GAS(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

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
        super(GAS,self).__init__('GAS')

        self.ar = ar
        self.sc = sc
        self.integ = integ
        self.param_no = self.ar + self.sc + 1
        self.max_lag = max(self.ar,self.sc)
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        self.skewness = False

        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data_original = self.data.copy()

        for order in range(0,self.integ):
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name

        self._create_model_matrices()
        self._create_parameters()

    def _bootstrap_scores(self,beta):
        """ Bootstraps the filtered series

        Returns
        ----------
        theta_sample : np.array
            sample of filtered series
        """     
        thetas,_,scores = self._model(beta)
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        theta_sample = np.ones(self.model_Y.shape[0])*parm[0]
        scores_sample = np.zeros(self.model_Y.shape[0])
        pseudo_theta = np.append(thetas,self.score_function(self.model_Y[-1],thetas[-1],model_scale,model_shape))
        sample_Y = self.draw_variable(self.link(pseudo_theta[1:]),model_scale,model_shape,self.model_Y.shape[0])

        for t in range(0,self.model_Y.shape[0]):
            if t < self.max_lag:
                theta_sample[t] = parm[0]/(1-np.sum(parm[1:(self.ar+1)]))
            else:
                theta_sample[t] += np.dot(parm[1:1+self.ar],theta_sample[(t-self.ar):t][::-1]) + np.dot(parm[1+self.ar:1+self.ar+self.sc],scores_sample[(t-self.sc):t][::-1])

            scores_sample[t] = self.score_function(sample_Y[t],self.link(theta_sample[t]),model_scale,model_shape, model_skewness)
        return theta_sample

    def _create_model_matrices(self):
        """ Creates model matrices/vectors

        Returns
        ----------
        None (changes model attributes)
        """

        self.model_Y = np.array(self.data[self.max_lag:self.data.shape[0]])
        self.model_scores = np.zeros(self.model_Y.shape[0])

    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        self.parameters.add_parameter('Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

        for ar_term in range(self.ar):
            self.parameters.add_parameter('AR(' + str(ar_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for sc_term in range(self.sc):
            self.parameters.add_parameter('SC(' + str(sc_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

    def _get_scale_and_shape(self,parm):
        """ Obtains appropriate model scale and shape parameters

        Parameters
        ----------
        parm : np.array
            Transformed parameter vector

        Returns
        ----------
        None (changes model attributes)
        """

        if self.scale is True:
            if self.shape is True:
                model_shape = parm[-1]  
                model_scale = parm[-2]
            else:
                model_shape = 0
                model_scale = parm[-1]
        else:
            model_scale = 0
            model_shape = 0 

        if self.skewness is True:
            model_skewness = parm[-3]
        else:
            model_skewness = 0

        return model_scale, model_shape, model_skewness

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

        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        theta = np.ones(self.model_Y.shape[0])*parm[0]
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        for t in range(0,self.model_Y.shape[0]):
            if t < self.max_lag:
                theta[t] = parm[0]/(1-np.sum(parm[1:(self.ar+1)]))
            else:
                theta[t] += np.dot(parm[1:1+self.ar],theta[(t-self.ar):t][::-1]) + np.dot(parm[1+self.ar:1+self.ar+self.sc],self.model_scores[(t-self.sc):t][::-1])

            self.model_scores[t] = self.score_function(self.model_Y[t],self.link(theta[t]),model_scale,model_shape,model_skewness)

        return theta, self.model_Y, self.model_scores

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

            Y_exp = np.append(Y_exp,[self.link(new_value)])
            theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
            scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

        return Y_exp

    def _sim_prediction(self,theta,Y,scores,h,t_params,simulations):
        """ Simulates a h-step ahead mean prediction

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

        simulations : int
            How many simulations to perform

        Returns
        ----------
        Matrix of simulations
        """     

        model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_params)

        sim_vector = np.zeros([simulations,h])

        for n in range(0,simulations):
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

                rnd_value = self.draw_variable(self.link(new_value),model_scale,model_shape,model_skewness,1)[0]
                Y_exp = np.append(Y_exp,[rnd_value])
                theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
                scores_exp = np.append(scores_exp,scores[np.random.randint(scores.shape[0])]) # expectation of score is zero

            sim_vector[n] = Y_exp[-h:]

        return np.transpose(sim_vector)


    def _summarize_simulations(self,mean_values,sim_vector,date_index,h,past_values):
        """ Summarizes a simulation vector and a mean vector of predictions

        Parameters
        ----------
        mean_values : np.array
            Mean predictions for h-step ahead forecasts

        sim_vector : np.array
            N simulation predictions for h-step ahead forecasts

        date_index : pd.DateIndex or np.array
            Dates for the simulations

        h : int
            How many steps ahead are forecast

        past_values : int
            How many past observations to include in the forecast plot

        intervals : Boolean
            Would you like to show prediction intervals for the forecast?
        """ 

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[(mean_values.shape[0]-h):(mean_values.shape[0])],0,0))
        forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index

    def plot_fit(self,intervals=False,**kwargs):
        """ Plots the fit of the model

        Notes
        ----------
        Intervals are bootstrapped as follows: take the filtered values from the
        algorithm (thetas). Use these thetas to generate a pseudo data stream from
        the measurement density. Use the GAS algorithm and estimated parameters to
        filter the pseudo data. Repeat this N times. 

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
                error_bars.append(np.array([np.percentile(i,5) for i in sim_vector]))
                error_bars.append(np.array([np.percentile(i,95) for i in sim_vector]))

            plt.figure(figsize=figsize)
            plt.subplot(2,1,1)
            plt.title("Model fit for " + self.data_name)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, error_bars[0], error_bars[1], alpha=0.15,label='95% C.I. for Bootstrapped GAS')    

            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,self.link(mu),label='GAS Filter',c='black')
            plt.legend(loc=2)   

            plt.subplot(2,1,2)
            plt.title("Filtered values for " + self.data_name)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, error_bars[0], error_bars[1], alpha=0.15,label='95% C.I. for Bootstrapped GAS')    

            plt.plot(date_index,self.link(mu),label='GAS Filter',c='black')
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
        data = self.data[-h:]

        plt.plot(predictions.index,data,label='Data')
        plt.plot(predictions.index,predictions,label='Predictions',c='black')
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

            theta, Y, scores = self._model(self.parameters.get_parameter_values())          
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            mean_values = self._mean_prediction(theta,Y,scores,h,t_params)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result