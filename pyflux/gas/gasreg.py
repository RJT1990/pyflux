import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices, dmatrix, demo_data

from ..parameter import Parameter, Parameters
from .. import inference as ifr
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
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        self.skewness = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.param_no = self.X.shape[1]
        self.y_name = self.y.design_info.describe()
        self.data_name = self.y_name
        self.X_names = self.X.design_info.describe().split(" + ")
        self.y = np.array([self.y]).ravel()
        self.data = self.y
        self.X = np.array([self.X])[0]
        self.index = data.index
        self.initial_values = np.zeros(self.param_no)

        self._create_model_matrices()
        self._create_parameters()

    def _bootstrap_scores(self,beta):
        """ Bootstraps the filtered series

        Returns
        ----------
        theta_sample : np.array
            sample of filtered series
        """     
        thetas,_,scores,coefficients = self._model(beta)
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        
        theta_sample = np.ones(self.model_Y.shape[0])
        scores_sample = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))
        coefficients_sample = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))
        coefficients_sample[:,0] = self.initial_values

        pseudo_theta = np.append(thetas,np.dot(self.X[-1],coefficients[:,-1]))
        sample_Y = self.draw_variable(self.link(pseudo_theta[1:]),model_scale,model_shape,model_skewness,self.model_Y.shape[0])

        for t in range(0,self.model_Y.shape[0]):
            theta_sample[t] = np.dot(self.X[t],coefficients_sample[:,t])
            scores_sample[:,t] = self.score_function(self.X[t],sample_Y[t],self.link(theta_sample[t]),model_scale,model_shape)
            coefficients_sample[:,t+1] = coefficients_sample[:,t] + parm[0:self.X.shape[1]]*scores_sample[:,t] 
        return theta_sample, coefficients_sample

    def _create_model_matrices(self):
        """ Creates model matrices/vectors

        Returns
        ----------
        None (changes model attributes)
        """

        self.model_Y = self.data
        self.model_scores = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))

    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        for parm in range(self.param_no):
            self.parameters.add_parameter('Scale ' + self.X_names[parm],ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

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
        coefficients = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))
        coefficients[:,0] = self.initial_values
        theta = np.zeros(self.model_Y.shape[0]+1)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        for t in range(0,self.model_Y.shape[0]):
            theta[t] = np.dot(self.X[t],coefficients[:,t])
            self.model_scores[:,t] = self.score_function(self.X[t],self.model_Y[t],self.link(theta[t]),model_scale,model_shape,model_skewness)
            coefficients[:,t+1] = coefficients[:,t] + parm[0:self.X.shape[1]]*self.model_scores[:,t] 
        return theta[:-1], self.model_Y, self.model_scores[:-1], coefficients

    def plot_fit(self,**kwargs):
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

            date_index = self.index.copy()
            mu, Y, scores, coefficients = self._model(self.parameters.get_parameter_values())

            plt.figure(figsize=figsize) 
            
            plt.subplot(len(self.X_names)+1, 1, 1)
            plt.title(self.y_name + " Filtered")
            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,self.link(mu),label='GAS Filter',c='black')
            plt.legend(loc=2)

            for coef in range(0,len(self.X_names)):
                plt.subplot(len(self.X_names)+1, 1, 2+coef)
                plt.title("Beta " + self.X_names[coef]) 
                plt.plot(date_index,coefficients[coef,0:-1],label='Coefficient')
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

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:

            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            X_pred = X_oos[:h]
            date_index = self.shift_dates(h)
            _, _, _, coefficients = self._model(self.parameters.get_parameter_values()) 
            coefficients_star = coefficients.T[-1]
            theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]
            t_params = self.transform_parameters()
            model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_params)

            # Measurement prediction intervals
            rnd_value = self.draw_variable(self.link(theta_pred),model_scale,model_shape,model_skewness,[1500,theta_pred.shape[0]])

            error_bars = []
            for pre in range(5,100,5):
                error_bars.append(np.insert([np.percentile(i,pre) for i in rnd_value.T] - self.link(theta_pred),0,0))

            plot_values = np.append(self.y,self.link(theta_pred))
            plot_values = plot_values[-h-past_values:]
            forecasted_values = np.append(self.y[-1],self.link(theta_pred))
            plot_index = date_index[-h-past_values:]

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])            
            
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.y_name)
            plt.xlabel("Time")
            plt.ylabel(self.y_name)
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

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:

            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            X_pred = X_oos[:h]
            date_index = self.shift_dates(h)
            _, _, _, coefficients = self._model(self.parameters.get_parameter_values()) 
            coefficients_star = coefficients.T[-1]
            theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]

            result = pd.DataFrame(self.link(theta_pred))
            result.rename(columns={0:self.y_name}, inplace=True)
            result.index = date_index[-h:]

            return result