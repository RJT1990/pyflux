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
from .gasmodels import *

class GASX(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GENERALIZED AUTOREGRESSIVE SCORE E(X)OGENOUS VARIABLE (GASX) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    formula : string
        patsy string describing the regression

    ar : int
        Field to specify how many AR lags the model will have.

    sc : int
        Field to specify how many score lags terms the model will have.

    integ : int (default : 0)
        Specifies how many time to difference the time series.

    dist : str
        Which distribution to use

    gradient_only : Boolean (default: True)
        If true, will only use gradient rather than second-order terms
        to construct the modified score.
    """

    def __init__(self,data,formula,ar,sc,family,integ=0,gradient_only=False):

        # Initialize TSM object     
        super(GASX,self).__init__('GASX')

        self.ar = ar
        self.sc = sc
        self.integ = integ
        self.gradient_only = gradient_only
        self.param_no = self.ar + self.sc
        self.max_lag = max(self.ar,self.sc)
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data.copy()
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.param_no = self.X.shape[1]
        self.y_name = self.y.design_info.describe()
        self.data_name = self.y_name
        self.X_names = self.X.design_info.describe().split(" + ")
        self.y = np.array([self.y]).ravel()
        self.data = self.y.copy()
        self.X = np.array([self.X])[0]
        self.index = data.index

        # Difference data
        for order in range(0,self.integ):
            self.y = np.diff(self.y)
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name

        self._create_model_matrices()
        self._create_parameters()

        self.family = family
        
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform = self.family.setup()
        self.model_name = self.model_name2 + "X(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ")"

        for no, i in enumerate(self.family.build_parameters()):
            self.parameters.add_parameter(i[0],i[1],i[2])
            self.parameters.parameter_list[no+self.ar+self.sc].start = i[3]

        self.parameters.parameter_list[0].start = self.mean_transform(np.mean(self.data))

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

        for ar_term in range(self.ar):
            self.parameters.add_parameter('AR(' + str(ar_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for sc_term in range(self.sc):
            self.parameters.add_parameter('SC(' + str(sc_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for parm in range(len(self.X_names)):
            self.parameters.add_parameter('Beta ' + self.X_names[parm],ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

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
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        theta = np.matmul(self.X[self.integ+self.max_lag:],parm[self.sc+self.ar:(self.sc+self.ar+len(self.X_names))])

        # Loop over time series
        for t in range(0,self.model_Y.shape[0]):
            if t < self.max_lag:
                theta[t] += parm[self.ar+self.sc]/(1-np.sum(parm[:self.ar]))
            else:
                theta[t] += np.dot(parm[:self.ar],theta[(t-self.ar):t][::-1]) + np.dot(parm[self.ar:self.ar+self.sc],self.model_scores[(t-self.sc):t][::-1])

            self.model_scores[t] = self.family.score_function(self.model_Y[t],self.link(theta[t]),model_scale,model_shape,model_skewness)

        return theta, self.model_Y, self.model_scores

    def _mean_prediction(self,theta,Y,scores,h,t_params,X_oos):
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

        X_oos : np.array
            Out of sample X data

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

            new_value += np.matmul(X_oos[t,:],t_params[self.sc+self.ar:(self.sc+self.ar+len(self.X_names))])            

            if self.model_name2 == "Exponential GAS":
                Y_exp = np.append(Y_exp,[1.0/self.link(new_value)])
            else:
                Y_exp = np.append(Y_exp,[self.link(new_value)])
            theta_exp = np.append(theta_exp,[new_value]) # For indexing consistency
            scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

        return Y_exp

    def _sim_prediction(self,theta,Y,scores,h,t_params,X_oos,simulations):
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

        X_oos : np.array
            Out of sample X data

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

                new_value += np.matmul(X_oos[t,:],t_params[self.sc+self.ar:(self.sc+self.ar+len(self.X_names))])            

                if self.model_name2 == "Exponential GAS":
                    rnd_value = self.family.draw_variable(1.0/self.link(new_value),model_scale,model_shape,model_skewness,1)[0]
                else:
                    rnd_value = self.family.draw_variable(self.link(new_value),model_scale,model_shape,model_skewness,1)[0]

                rnd_value = self.family.draw_variable(self.link(new_value),model_scale,model_shape,model_skewness,1)[0]
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

    def neg_loglik(self,beta):
        theta, Y, _ = self._model(beta)
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y,self.link(theta),model_scale,model_shape,model_skewness)

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
            plt.figure(figsize=figsize)
            plt.subplot(2,1,1)
            plt.title("Model fit for " + self.data_name)

            if self.model_name2 == "Exponential GAS":
                values_to_plot = 1.0/self.link(mu)
            else:
                values_to_plot = self.link(mu)

            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,values_to_plot,label='GAS Filter',c='black')
            plt.legend(loc=2)   

            plt.subplot(2,1,2)
            plt.title("Filtered values for " + self.data_name)
            plt.plot(date_index,values_to_plot,label='GAS Filter',c='black')
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

            # Retrieve data, dates and (transformed) parameters
            theta, Y, scores = self._model(self.parameters.get_parameter_values())          
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(theta,Y,scores,h,t_params,X_pred)
            sim_values = self._sim_prediction(theta,Y,scores,h,t_params,X_pred,15000)
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
            data1 = self.data_original.iloc[:-(h+t),:]
            data2 = self.data_original.iloc[-h+t:,:]
            x = GASX(ar=self.ar,sc=self.sc,formula=self.formula,integ=self.integ,family=self.family,data=self.data_original[:-h+t],gradient_only=self.gradient_only)
            x.fit(printer=False)
            if t == 0:
                predictions = x.predict(1,oos_data=data2)
            else:
                predictions = pd.concat([predictions,x.predict(h=1,oos_data=data2)])
        
        predictions.rename(columns={0:self.y_name}, inplace=True)
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
        predictions = self.predict_is(h)
        data = self.data[-h:]

        plt.plot(predictions.index,data,label='Data')
        plt.plot(predictions.index,predictions,label='Predictions',c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def predict(self,h=5, oos_data=None):
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
            theta, Y, scores = self._model(self.parameters.get_parameter_values())          
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            mean_values = self._mean_prediction(theta,Y,scores,h,t_params,X_pred)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result