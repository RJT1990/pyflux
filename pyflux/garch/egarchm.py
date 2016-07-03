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
from .. import gas as gas
from .. import data_check as dc

class EGARCHM(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** BETA-t-EGARCH IN MEAN MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    p : int
        Field to specify how many GARCH terms the model will have.

    q : int
        Field to specify how many SCORE terms the model will have.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.
    """

    def __init__(self,data,p,q,target=None):

        # Initialize TSM object
        super(EGARCHM,self).__init__('EGARCHM')

        # Parameters
        self.p = p
        self.q = q
        self.param_no = self.p + self.q + 2
        self.max_lag = max(self.p,self.q)
        self.leverage = False
        self.model_name = "EGARCHM(" + str(self.p) + "," + str(self.q) + ")"
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

        self.parameters.add_parameter('Vol Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

        for p_term in range(self.p):
            self.parameters.add_parameter('p(' + str(p_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        for q_term in range(self.q):
            self.parameters.add_parameter('q(' + str(q_term+1) + ')',ifr.Normal(0,0.5,transform=None),dst.q_Normal(0,3))

        self.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('Returns Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))
        self.parameters.add_parameter('GARCH-M',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))

        # Starting values
        self.parameters.parameter_list[0].start = np.log(0.02)
        
        for i in range(1,self.p+1):
            if self.p > 1:
                self.parameters.parameter_list[i].start = 0.0001
            elif self.p == 1:
                self.parameters.parameter_list[i].start = 0.95

        for i in range(1+self.q,self.p+1+self.q):
            if self.q > 1:
                self.parameters.parameter_list[i].start = 0.0001
            elif self.q == 1:
                self.parameters.parameter_list[i].start = 0.01

        self.parameters.parameter_list[-3].start = 2.0

    def _model(self,beta):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        lambda : np.array
            Contains the values for the conditional volatility series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        scores : np.array
            Contains the score terms for the time series
        """

        Y = np.array(self.data[self.max_lag:self.data.shape[0]])
        X = np.ones(Y.shape[0])
        scores = np.zeros(Y.shape[0])

        # Transform parameters
        parm = np.array([self.parameters.parameter_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        lmda = np.ones(Y.shape[0])*parm[0]

        # Loop over time series
        for t in range(0,Y.shape[0]):

            if t < self.max_lag:

                lmda[t] = parm[0]/(1-np.sum(parm[1:(self.p+1)]))

            else:

                # Loop over GARCH terms
                for p_term in range(0,self.p):
                    lmda[t] += parm[1+p_term]*lmda[t-p_term-1]

                # Loop over Score terms
                for q_term in range(0,self.q):
                    lmda[t] += parm[1+self.p+p_term]*scores[t-q_term-1]

                if self.leverage is True:
                    lmda[t] += parm[-4]*np.sign(-(Y[t-1]-parm[-2]-parm[-1]*np.exp(lmda[t-1])))*(scores[t-1]+1)

            scores[t] = gas.BetatScore.mu_adj_score(Y[t]-parm[-2]-parm[-1]*np.exp(lmda[t]),0,lmda[t],parm[-3])

        return lmda, Y, scores

    def _mean_prediction(self,lmda,Y,scores,h,t_params):
        """ Creates a h-step ahead mean prediction

        Parameters
        ----------
        lmda : np.array
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
        lmda_exp = lmda.copy()
        scores_exp = scores.copy()
        Y_exp = Y.copy()

        # Loop over h time periods          
        for t in range(0,h):
            new_value = t_params[0]

            if self.p != 0:
                for j in range(1,self.p+1):
                    new_value += t_params[j]*lmda_exp[-j]

            if self.q != 0:
                for k in range(1,self.q+1):
                    new_value += t_params[k+self.p]*scores_exp[-k]

            if self.leverage is True:
                new_value += t_params[1+self.p+self.q]*np.sign(-(Y_exp[scores_exp.shape[0]-1]-t_params[-2]-t_params[-1]*np.exp(lmda_exp[-1])))*(scores_exp[-1]+1)

            lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
            scores_exp = np.append(scores_exp,[0]) # expectation of score is zero
            Y_exp = np.append(Y_exp,[0])

        return lmda_exp

    def _sim_prediction(self,lmda,Y,scores,h,t_params,simulations):
        """ Simulates a h-step ahead mean prediction

        Parameters
        ----------
        lmda : np.array
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
            lmda_exp = lmda.copy()
            scores_exp = scores.copy()
            Y_exp = Y.copy()

            # Loop over h time periods          
            for t in range(0,h):
                new_value = t_params[0]

                if self.p != 0:
                    for j in range(1,self.p+1):
                        new_value += t_params[j]*lmda_exp[-j]

                if self.q != 0:
                    for k in range(1,self.q+1):
                        new_value += t_params[k+self.p]*scores_exp[-k]

                if self.leverage is True:
                    new_value += t_params[1+self.p+self.q]*np.sign(-(Y_exp[scores_exp.shape[0]-1]-t_params[-2]-t_params[-1]*np.exp(lmda_exp[-1])))*(scores_exp[-1]+1)

                lmda_exp = np.append(lmda_exp,[new_value]) # For indexing consistency
                scores_exp = np.append(scores_exp,scores[np.random.randint(scores.shape[0])]) # expectation of score is zero
                Y_exp = np.append(Y_exp,Y[np.random.randint(Y.shape[0])]) # bootstrap returns

            sim_vector[n] = lmda_exp[-h:]

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

    def add_leverage(self):
        """ Adds leverage term to the model

        Returns
        ----------
        None (changes instance attributes)
        """             

        if self.leverage is True:
            pass
        else:
            self.leverage = True
            self.param_no += 1
            self.parameters.parameter_list.pop()            
            self.parameters.parameter_list.pop()
            self.parameters.parameter_list.pop()
            self.parameters.add_parameter('Leverage Term',ifr.Uniform(transform=None),dst.q_Normal(0,3))
            self.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
            self.parameters.add_parameter('Returns Constant',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))
            self.parameters.add_parameter('GARCH-M',ifr.Normal(0,3,transform=None),dst.q_Normal(0,3))
            self.parameters.parameter_list[-3].start = 2.0

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

        lmda, Y, ___ = self._model(beta)
        loc = np.ones(lmda.shape[0])*self.parameters.parameter_list[-2].prior.transform(beta[-2]) + self.parameters.parameter_list[-1].prior.transform(beta[-1])*np.exp(lmda)
        return -np.sum(ss.t.logpdf(x=Y,
            df=self.parameters.parameter_list[-3].prior.transform(beta[-3]),
            loc=loc,scale=np.exp(lmda/2.0)))
    
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
            t_params = self.transform_parameters()
            plt.figure(figsize=figsize)
            date_index = self.index[max(self.p,self.q):]
            sigma2, Y, ___ = self._model(self.parameters.get_parameter_values())
            plt.plot(date_index,np.abs(Y-t_params[-2]-t_params[-1]*np.exp(sigma2)),label=self.data_name + ' Absolute Values')
            plt.plot(date_index,np.exp(sigma2/2.0),label='EGARCH(' + str(self.p) + ',' + str(self.q) + ') std',c='black')                   
            plt.title(self.data_name + " Volatility Plot")  
            plt.legend(loc=2)   
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
            lmda, Y, scores = self._model(self.parameters.get_parameter_values())           
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(lmda,Y,scores,h,t_params)
            sim_values = self._sim_prediction(lmda,Y,scores,h,t_params,15000)
            error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values,sim_values,date_index,h,past_values)

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], np.exp((forecasted_values-pre)/2), np.exp((forecasted_values+pre)/2),alpha=alpha[count])            
            
            plt.plot(plot_index,np.exp(plot_values/2.0))
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
            x = EGARCH(p=self.p,q=self.q,data=self.data[:-h+t])
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

            sigma2, Y, scores = self._model(self.parameters.get_parameter_values()) 
            date_index = self.shift_dates(h)
            t_params = self.transform_parameters()

            mean_values = self._mean_prediction(sigma2,Y,scores,h,t_params)
            forecasted_values = mean_values[-h:]
            result = pd.DataFrame(np.exp(forecasted_values/2.0))
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result


