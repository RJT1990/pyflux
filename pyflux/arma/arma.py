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

from .arma_recursions import arima_recursion

class ARIMA(tsm.TSM):
    """ Inherits time series methods from TSM parent class.

    **** AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Field to specify the univariate time series data that will be used.

    ar : int
        Field to specify how many AR lags the model will have.

    ma : int
        Field to specify how many MA lags the model will have.

    integ : int (default : 0)
        Specifies how many times to difference the time series.

    target : str (if data is a pd.DataFrame) or int (if data is a np.ndarray)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.
    """

    def __init__(self, data, ar, ma, integ=0, target=None):

        # Initialize TSM object
        super(ARIMA, self).__init__('ARIMA')

        # Latent Variable information
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.model_name = "ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ")"
        self.z_no = self.ar + self.ma + 2
        self.max_lag = max(self.ar,self.ma)
        self._z_hide = 0 # Whether to cutoff variance latent variables from results
        self.supported_methods = ["MLE", "PML", "Laplace", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data = self.data.astype(np.float) # treat as float for Cython
        self.data_original = self.data.copy()

        # Difference data
        for order in range(0, self.integ):
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name

        self.X = self._ar_matrix()
        self._create_latent_variables()

    def _ar_matrix(self):
        """ Creates Autoregressive Matrix

        Returns
        ----------
        X : np.ndarray
            Autoregressive Matrix

        """
        Y = np.array(self.data[self.max_lag:self.data.shape[0]])
        X = np.ones(Y.shape[0])

        if self.ar != 0:
            for i in range(0,self.ar):
                X = np.vstack((X,self.data[(self.max_lag-i-1):-i-1]))

        return X

    def _create_latent_variables(self):
        """ Creates the model's latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        self.latent_variables.add_z('Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

        for ar_term in range(self.ar):
            self.latent_variables.add_z('AR(' + str(ar_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for ma_term in range(self.ma):
            self.latent_variables.add_z('MA(' + str(ma_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        self.latent_variables.add_z('Sigma',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        self.latent_variables.z_list[0].start = np.mean(self.data)

    def _model(self, beta):
        """ Creates the structure of the model (model matrices etc)

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        Y = np.array(self.data[self.max_lag:])

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(self.X),z[0:-1-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion(z, mu, Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _mean_prediction(self, mu, Y, h, t_z):
        """ Creates a h-step ahead mean prediction

        Parameters
        ----------
        mu : np.ndarray
            The past predicted values

        Y : np.ndarray
            The past data

        h : int
            How many steps ahead for the prediction

        t_z : np.ndarray
            A vector of (transformed) latent variables

        Returns
        ----------
        h-length vector of mean predictions
        """     

        # Create arrays to iteratre over
        Y_exp = Y.copy()
        mu_exp = mu.copy()

        # Loop over h time periods          
        for t in range(0,h):
            new_value = t_z[0]

            if self.ar != 0:
                for j in range(1,self.ar+1):
                    new_value += t_z[j]*Y_exp[-j]

            if self.ma != 0:
                for k in range(1,self.ma+1):
                    if (k-1) >= t:
                        new_value += t_z[k+self.ar]*(Y_exp[-k]-mu_exp[-k])

            Y_exp = np.append(Y_exp,[new_value])
            mu_exp = np.append(mu_exp,[0]) # For indexing consistency

        return Y_exp

    def _sim_prediction(self, mu, Y, h, t_z, simulations):
        """ Simulates a h-step ahead mean prediction

        Parameters
        ----------
        mu : np.ndarray
            The past predicted values

        Y : np.ndarray
            The past data

        h : int
            How many steps ahead for the prediction

        t_params : np.ndarray
            A vector of (transformed) latent variables

        simulations : int
            How many simulations to perform

        Returns
        ----------
        Matrix of simulations
        """     

        sim_vector = np.zeros([simulations,h])

        for n in range(0, simulations):
            # Create arrays to iteratre over        
            Y_exp = Y.copy()
            mu_exp = mu.copy()

            # Loop over h time periods          
            for t in range(0,h):

                new_value = t_z[0] + np.random.randn(1)*t_z[-1]

                if self.ar != 0:
                    for j in range(1, self.ar+1):
                        new_value += t_z[j]*Y_exp[-j]

                if self.ma != 0:
                    for k in range(1, self.ma+1):
                        if (k-1) >= t:
                            new_value += t_z[k+self.ar]*(Y_exp[-k]-mu_exp[-k])

                Y_exp = np.append(Y_exp,[new_value])
                mu_exp = np.append(mu_exp,[0]) # For indexing consistency

                sim_vector[n] = Y_exp[-h:]

        return np.transpose(sim_vector)

    def _summarize_simulations(self,mean_values,sim_vector,date_index,h,past_values):
        """ Produces simulation forecasted values and prediction intervals

        Parameters
        ----------
        mean_values : np.ndarray
            Mean predictions for h-step ahead forecasts

        sim_vector : np.ndarray
            N simulation predictions for h-step ahead forecasts

        date_index : pd.DateIndex or np.ndarray
            Date index for the simulation

        h : int
            How many steps ahead are forecast

        past_values : int
            How many past observations to include in the forecast plot

        intervals : boolean
            Would you like to show prediction intervals for the forecast?
        """         

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[-h:],0,0))
        forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index
        
    def neg_loglik(self, beta):
        """ Calculates the negative log-likelihood of the model

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._model(beta)
        return -np.sum(ss.norm.logpdf(Y, loc=mu, scale=self.latent_variables.z_list[-1].prior.transform(beta[-1])))

    def plot_fit(self, **kwargs):
        """ 
        Plots the fit of the model
        """

        figsize = kwargs.get('figsize',(10,7))

        plt.figure(figsize=figsize)
        date_index = self.index[max(self.ar,self.ma):self.data.shape[0]]
        mu, Y = self._model(self.latent_variables.get_z_values())
        plt.plot(date_index,Y,label='Data')
        plt.plot(date_index,mu,label='Filter',c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def plot_predict(self, h=5, past_values=20, intervals=True, **kwargs):
        """ Plots forecasts with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : boolean
            Would you like to show prediction intervals for the forecast?

        Returns
        ----------
        - Plot of the forecast
        """     

        figsize = kwargs.get('figsize',(10,7))

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            # Retrieve data, dates and (transformed) latent variables
            mu, Y = self._model(self.latent_variables.get_z_values())         
            date_index = self.shift_dates(h)
            t_z = self.transform_z()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(mu,Y,h,t_z)
            sim_values = self._sim_prediction(mu,Y,h,t_z,15000)
            error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])            
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
            plt.show()

    def predict_is(self, h=5):
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
            x = ARIMA(ar=self.ar,ma=self.ma,integ=self.integ,data=self.data_original[:-h+t])
            x.fit(printer=False)
            if t == 0:
                predictions = x.predict(1)
            else:
                predictions = pd.concat([predictions,x.predict(1)])
        
        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

    def plot_predict_is(self, h=5, **kwargs):
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

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:

            mu, Y = self._model(self.latent_variables.get_z_values())         
            date_index = self.shift_dates(h)
            t_z = self.transform_z()

            mean_values = self._mean_prediction(mu,Y,h,t_z)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result