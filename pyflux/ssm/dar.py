import copy
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from patsy import dmatrices, dmatrix, demo_data

from .. import families as fam
from .. import tsm as tsm
from .. import data_check as dc

from .kalman import *

class DAR(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** DYNAMIC AUTOREGRESSIVE MODEL ****

    Parameters
    ----------

    ar : int
        Number of autoregressive lags

    data : pd.DataFrame
        Field to specify the data that will be used
    """

    def __init__(self, data, ar, integ=0, target=None):

        # Initialize TSM object
        super(DAR, self).__init__('DAR')

        # Latent Variable information
        self.ar = ar
        self.integ = integ
        self.target = target
        self.model_name = "DAR(" + str(self.ar) + ", integrated=" + str(self.integ) + ")"
        self.max_lag = self.ar
        self._z_hide = 0 # Whether to cutoff latent variables from results table
        self.supported_methods = ["MLE", "PML", "Laplace", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.data_original = data.copy()
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data = self.data.astype(np.float) # treat as float for Cython
        self.data_original_nondf = self.data.copy()

        # Difference data
        for order in range(0, self.integ):
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name

        self.X = self._ar_matrix()
        self.data = self.data[self.max_lag:]
        self.y = self.data
        self.y_name = self.data_name
        self._create_latent_variables()
        self.z_no = len(self.latent_variables.z_list)

    def _ar_matrix(self):
        """ Creates Autoregressive matrix

        Returns
        ----------
        X : np.ndarray
            Autoregressive Matrix

        """
        Y = np.array(self.data[self.max_lag:self.data.shape[0]])
        X = np.ones(Y.shape[0])

        if self.ar != 0:
            for i in range(0, self.ar):
                X = np.vstack((X,self.data[(self.max_lag-i-1):-i-1]))

        return X.T

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        self.latent_variables.add_z('Sigma^2 irregular', fam.Flat(transform='exp'), fam.Normal(0,3))

        self.latent_variables.add_z('Constant', fam.Flat(transform=None), fam.Normal(0,3))

        for parm in range(1,self.ar+1):
            self.latent_variables.add_z('Sigma^2 AR(' + str(parm) + ')', fam.Flat(transform='exp'), fam.Normal(0,3))

    def _forecast_model(self,beta,Z,h):
        """ Creates forecasted states and variances

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        Returns
        ----------
        a : np.ndarray
            Forecasted states

        P : np.ndarray
            Variance of forecasted states
        """     

        T, _, R, Q, H = self._ss_matrices(beta)
        return dl_univariate_kalman_fcst(self.data,Z,H,T,Q,R,0.0,h)

    def _model(self,data,beta):
        """ Creates the structure of the model

        Parameters
        ----------
        data : np.array
            Contains the time series

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        a,P,K,F,v : np.array
            Filted states, filtered variances, Kalman gains, F matrix, residuals
        """     

        T, Z, R, Q, H = self._ss_matrices(beta)

        return dl_univariate_kalman(data,Z,H,T,Q,R,0.0)

    def _ss_matrices(self,beta):
        """ Creates the state space matrices required

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        T, Z, R, Q, H : np.array
            State space matrices used in KFS algorithm
        """     

        T = np.identity(self.z_no-1)
        H = np.identity(1)*self.latent_variables.z_list[0].prior.transform(beta[0])       
        Z = self.X
        R = np.identity(self.z_no-1)
        
        Q = np.identity(self.z_no-1)
        for i in range(0,self.z_no-1):
            Q[i][i] = self.latent_variables.z_list[i+1].prior.transform(beta[i+1])

        return T, Z, R, Q, H

    def neg_loglik(self,beta):
        """ Creates the negative log marginal likelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative log logliklihood of the model
        """         
        _, _, _, F, v = self._model(self.y,beta)
        loglik = 0.0
        for i in range(0,self.y.shape[0]):
            loglik += np.linalg.slogdet(F[:,:,i])[1] + np.dot(v[i],np.dot(np.linalg.pinv(F[:,:,i]),v[i]))
        return -(-((self.y.shape[0]/2)*np.log(2*np.pi))-0.5*loglik.T[0].sum())

    def plot_predict(self, h=5, past_values=20, intervals=True, **kwargs):        
        """ Makes forecast with the estimated model

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
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            y_holder = self.y.copy() # holds past data and predicted data to create AR matrix
            full_X = self.X.copy()
            full_X = np.append(full_X,np.array([np.append(1.0, y_holder[-self.ar:][::-1])]), axis=0)
            Z = full_X

            # Construct Z matrix
            for step in range(h):
                a, P = self._forecast_model(self.latent_variables.get_z_values(),Z,step)
                new_value = np.dot(Z[-1,:],a[:,self.y.shape[0]+step])
                y_holder = np.append(y_holder, new_value)
                Z = np.append(Z, np.array([np.append(1.0, y_holder[-self.ar:][::-1])]), axis=0)

            # Retrieve data, dates and (transformed) latent variables         
            a, P = self._forecast_model(self.latent_variables.get_z_values(),Z,h)
            smoothed_series = np.zeros(self.y.shape[0]+h)
            series_variance = np.zeros(self.y.shape[0]+h)
            for t in range(self.y.shape[0]+h):
                smoothed_series[t] = np.dot(Z[t],a[:,t])
                series_variance[t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + self.latent_variables.z_list[0].prior.transform(self.latent_variables.get_z_values()[0])    

            date_index = self.shift_dates(h)
            plot_values = smoothed_series[-h-past_values:]
            forecasted_values = smoothed_series[-h:]
            lower = forecasted_values - 1.98*np.power(series_variance[-h:],0.5)
            upper = forecasted_values + 1.98*np.power(series_variance[-h:],0.5)
            lower = np.append(plot_values[-h-1],lower)
            upper = np.append(plot_values[-h-1],upper)

            plot_index = date_index[-h-past_values:]

            plt.figure(figsize=figsize)
            if intervals == True:
                plt.fill_between(date_index[-h-1:], lower, upper, alpha=0.2)            

            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.y_name)
            plt.xlabel("Time")
            plt.ylabel(self.y_name)
            plt.show()

    def plot_fit(self,intervals=False,**kwargs):
        """ Plots the fit of the model

        Parameters
        ----------
        intervals : Boolean
            Whether to plot 95% confidence interval of states

        Returns
        ----------
        None (plots data and the fit)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))
        series_type = kwargs.get('series_type','Smoothed')

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            date_index = copy.deepcopy(self.index)
            date_index = date_index[self.integ+self.ar:]

            if series_type == 'Smoothed':
                mu, V = self.smoothed_state(self.data,self.latent_variables.get_z_values())
            elif series_type == 'Filtered':
                mu, V, _, _, _ = self._model(self.data,self.latent_variables.get_z_values())
            else:
                mu, V = self.smoothed_state(self.data,self.latent_variables.get_z_values())

            # Create smoothed/filtered aggregate series
            _, Z, _, _, _ = self._ss_matrices(self.latent_variables.get_z_values())
            smoothed_series = np.zeros(self.y.shape[0])

            for t in range(0,self.y.shape[0]):
                smoothed_series[t] = np.dot(Z[t],mu[:,t])

            plt.figure(figsize=figsize) 
            
            plt.subplot(self.z_no+1, 1, 1)
            plt.title(self.y_name + " Raw and " + series_type)  

            plt.plot(date_index,self.data,label='Data')
            plt.plot(date_index,smoothed_series,label=series_type,c='black')
            plt.legend(loc=2)

            for coef in range(0,self.z_no-1):
                V_coef = V[0][coef][:-1]    
                plt.subplot(self.z_no+1, 1, 2+coef)
                plt.title("Beta " + self.latent_variables.z_list[1+coef].name) 

                if intervals == True:
                    alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                    plt.fill_between(date_index[5:], mu[coef,0:mu.shape[1]-1][5:] + 1.98*np.sqrt(V_coef[5:]), mu[coef,0:mu.shape[1]-1][5:] - 1.98*np.sqrt(V_coef[5:]), alpha=0.15,label='95% C.I.') 
                plt.plot(date_index,mu[coef,0:mu.shape[1]-1],label='Data')
                plt.legend(loc=2)               
            
            plt.subplot(self.z_no+1, 1, self.z_no+1)
            plt.title("Measurement Error")
            plt.plot(date_index,self.data-smoothed_series,label='Irregular')
            plt.legend(loc=2)   

            plt.show()  

    def predict(self, h=5):        
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predictions
        """     

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            y_holder = self.y.copy() # holds past data and predicted data to create AR matrix
            full_X = self.X.copy()
            full_X = np.append(full_X,np.array([np.append(1.0, y_holder[-self.ar:][::-1])]), axis=0)
            Z = full_X

            for step in range(h):
                a, P = self._forecast_model(self.latent_variables.get_z_values(),Z,step)
                new_value = np.dot(Z[-1,:],a[:,self.y.shape[0]+step])
                y_holder = np.append(y_holder, new_value)
                Z = np.append(Z, np.array([np.append(1.0, y_holder[-self.ar:][::-1])]), axis=0)

            date_index = self.shift_dates(h)

            result = pd.DataFrame(y_holder[-h:])
            result.rename(columns={0:self.y_name}, inplace=True)
            result.index = date_index[-h:]

            return result

    def predict_is(self, h=5, fit_once=True):
        """ Makes dynamic in-sample predictions with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        fit_once : boolean
            (default: True) Fits only once before the in-sample prediction; if False, fits after every new datapoint

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        predictions = []

        for t in range(0,h):
            data1 = self.data_original_nondf[:-h+t]
            x = DAR(data=data1, ar=self.ar, integ=self.integ)
            if fit_once is False:
                x.fit(printer=False)
            if t == 0:
                if fit_once is True:
                    x.fit(printer=False)
                    saved_lvs = x.latent_variables
                predictions = x.predict(1)
            else:
                if fit_once is True:
                    x.latent_variables = saved_lvs
                predictions = pd.concat([predictions,x.predict(1)])

        predictions.rename(columns={0:self.y_name}, inplace=True)
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
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        plt.figure(figsize=figsize)
        predictions = self.predict_is(h)
        data = self.data[-h:]
        plt.plot(predictions.index,data,label='Data')
        plt.plot(predictions.index,predictions,label='Predictions',c='black')
        plt.title(self.y_name)
        plt.legend(loc=2)   
        plt.show()          

    def simulation_smoother(self,beta):
        """ Koopman's simulation smoother - simulates from states given
        model parameters and observations

        Parameters
        ----------

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - A simulated state evolution
        """         

        T, Z, R, Q, H = self._ss_matrices(beta)

        # Generate e_t+ and n_t+
        rnd_h = np.random.normal(0,np.sqrt(H),self.data.shape[0]+1)
        q_dist = ss.multivariate_normal([0.0, 0.0], Q)
        rnd_q = q_dist.rvs(self.data.shape[0]+1)

        # Generate a_t+ and y_t+
        a_plus = np.zeros((T.shape[0],self.data.shape[0]+1)) 
        a_plus[0,0] = np.mean(self.data[0:5])
        y_plus = np.zeros(self.data.shape[0])

        for t in range(0,self.data.shape[0]+1):
            if t == 0:
                a_plus[:,t] = np.dot(T,a_plus[:,t]) + rnd_q[t,:]
                y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]
            else:
                if t != self.data.shape[0]:
                    a_plus[:,t] = np.dot(T,a_plus[:,t-1]) + rnd_q[t,:]
                    y_plus[t] = np.dot(Z,a_plus[:,t]) + rnd_h[t]

        alpha_hat, _ = self.smoothed_state(self.data,beta)
        alpha_hat_plus, _ = self.smoothed_state(y_plus,beta)
        alpha_tilde = alpha_hat - alpha_hat_plus + a_plus
    
        return alpha_tilde

    def smoothed_state(self,data,beta):
        """ Creates the negative log marginal likelihood of the model

        Parameters
        ----------

        data : np.array
            Data to be smoothed

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - Smoothed states
        """         

        T, Z, R, Q, H = self._ss_matrices(beta)
        alpha, V = dl_univariate_KFS(data,Z,H,T,Q,R,0.0)
        return alpha, V

