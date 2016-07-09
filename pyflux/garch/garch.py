import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

class GARCH(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSCEDASTICITY (GARCH) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    p : int
        Field to specify how many GARCH terms the model will have.

    q : int
        Field to specify how many ARCH terms the model will have.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.
    """

    def __init__(self,data,p,q,target=None):

        # Initialize TSM object
        super(GARCH,self).__init__('GARCH')

        # Parameters
        self.p = p
        self.q = q
        self.param_no = self.p + self.q + 2
        self.max_lag = max(self.p,self.q)
        self.model_name = "GARCH(" + str(self.p) + "," + str(self.q) + ")"
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self._create_parameters()
        
    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        self.parameters.add_parameter('Vol Constant',ifr.Normal(0,3,transform='exp'),dst.q_Normal(0,3))

        for q_term in range(self.q):
            self.parameters.add_parameter('q(' + str(q_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for p_term in range(self.p):
            self.parameters.add_parameter('p(' + str(p_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))
        
        self.parameters.add_parameter('Returns Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

        self.parameters.parameter_list[0].start = self.parameters.parameter_list[0].prior.itransform(np.mean(np.power(self.data-np.mean(self.data),2)))
        
        for k in range(1,len(self.parameters.parameter_list)-1):
            self.parameters.parameter_list[k].start = 0.00001

    def _model(self,beta):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        sigma2 : np.array
            Contains the values for the conditional volatility series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        eps : np.array
            Contains the squared residuals (ARCH terms) for the time series
        """

        # Transform parameters
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        xeps = np.power(self.data-parm[-1],2)
        Y = np.array(self.data[self.max_lag:])
        eps = np.power(Y-parm[-1],2)
        X = np.ones(Y.shape[0])

        # ARCH terms
        if self.q != 0:
            for i in range(0,self.q):   
                X = np.vstack((X,xeps[(self.max_lag-i-1):-i-1]))
            sigma2 = np.matmul(np.transpose(X),parm[0:-self.p-1])
        else:
            sigma2 = np.transpose(X*parm[0])

        # GARCH terms
        if self.p != 0:
            for t in range(0,Y.shape[0]):
                if t < self.max_lag:
                    sigma2[t] = parm[0]/(1-np.sum(parm[(self.q+1):(self.q+self.p+1)]))
                elif t >= self.max_lag:
                    for k in range(0,self.p):
                        sigma2[t] += parm[1+self.q+k]*(sigma2[t-1-k])

        return sigma2, Y, eps

    def _mean_prediction(self,sigma2,Y,scores,h,t_params):
        """ Creates a h-step ahead mean prediction

        Parameters
        ----------
        sigma2 : np.array
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
        h-length vector of mean predictions
        """     

        # Create arrays to iteratre over
        sigma2_exp = sigma2.copy()
        scores_exp = scores.copy()

        # Loop over h time periods          
        for t in range(0,h):
            new_value = t_params[0]

            # ARCH
            if self.q != 0:
                for j in range(1,self.q+1):
                    new_value += t_params[j]*scores_exp[-j]

            # GARCH
            if self.p != 0:
                for k in range(1,self.p+1):
                    new_value += t_params[k+self.q]*sigma2_exp[-k]                  

            sigma2_exp = np.append(sigma2_exp,[new_value]) # For indexing consistency
            scores_exp = np.append(scores_exp,[0]) # expectation of score is zero

        return sigma2_exp

    def _sim_prediction(self,sigma2,Y,scores,h,t_params,simulations):
        """ Simulates a h-step ahead mean prediction

        Parameters
        ----------
        sigma2 : np.array
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
        sim_vector = np.zeros([simulations,h])

        for n in range(0,simulations):
            # Create arrays to iteratre over        
            sigma2_exp = sigma2.copy()
            scores_exp = scores.copy()

            # Loop over h time periods          
            for t in range(0,h):
                new_value = t_params[0]

                if self.q != 0:
                    for j in range(1,self.q+1):
                        new_value += t_params[j]*scores_exp[-j]

                if self.p != 0:
                    for k in range(1,self.p+1):
                        new_value += t_params[k+self.q]*sigma2_exp[-k]  

                sigma2_exp = np.append(sigma2_exp,[new_value]) # For indexing consistency
                scores_exp = np.append(scores_exp,scores[np.random.randint(scores.shape[0])]) # expectation of score is zero

            sim_vector[n] = sigma2_exp[-h:]
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
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[-h:],0,0))
        forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index

    def neg_loglik(self,beta):
        """ Creates the negative log-likelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        The negative logliklihood of the model
        """     

        sigma2, Y, __ = self._model(beta)
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        return -np.sum(ss.norm.logpdf(Y,loc=parm[-1]*np.ones(sigma2.shape[0]),scale=np.sqrt(sigma2)))

    def plot_fit(self,**kwargs):
        """ Plots the fit of the model

        Returns
        ----------
        None (plots data and the fit)
        """

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:
            plt.figure(figsize=figsize)
            date_index = self.index[max(self.p,self.q):]
            t_params = self.transform_parameters()
            sigma2, Y, ___ = self._model(self.parameters.get_parameter_values())
            plt.plot(date_index,np.abs(Y-t_params[-1]),label=self.data_name + ' Absolute Demeaned Values')
            plt.plot(date_index,np.power(sigma2,0.5),label='GARCH(' + str(self.p) + ',' + str(self.q) + ') Conditional Volatility',c='black')
            plt.title(self.data_name + " Volatility Plot")  
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
            sigma2, Y, scores = self._model(self.parameters.get_parameter_values())         
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(sigma2,Y,scores,h,t_params)
            sim_values = self._sim_prediction(sigma2,Y,scores,h,t_params,15000)
            error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])            
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name + " Conditional Volatility")
            plt.xlabel("Time")
            plt.ylabel(self.data_name + " Conditional Volatility")
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
            x = GARCH(p=self.p,q=self.q,data=self.data[0:-h+t])
            x.fit(printer=False)
            if t == 0:
                predictions = x.predict(1)
            else:
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

        plt.plot(date_index,np.abs(data),label='Data')
        plt.plot(date_index,np.power(predictions,0.5),label='Predictions',c='black')
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

            sigma2, Y, scores = self._model(self.parameters.get_parameter_values())         
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            mean_values = self._mean_prediction(sigma2,Y,scores,h,t_params)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result


