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

class GASskewt(GAS):
    """ Inherits GAS methods from GAS class (and time series methods from TSM class).

    **** skewt GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

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

    def __init__(self,data,ar,sc,integ=0,target=None,gradient_only=True):

        # Initialize TSM object     
        super(GASskewt,self).__init__(data=data,ar=ar,sc=sc,integ=integ,
            target=target,gradient_only=gradient_only)

        self.model_name = "skewt GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") REGRESSION"     
        self.dist = 'skewt'
        self.link = np.array
        self.scale = True
        self.shape = True
        self.skewness = True
        self.param_no += 3
        self.parameters.add_parameter('Skewness',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.parameter_list[0].start = np.mean(self.data)
        self.parameters.parameter_list[-1].start = 2.0

        if gradient_only is False:
            self.score_function = self.adj_score_function
        else:
            self.score_function = self.default_score_function

    def adj_score_function(self,y,mean,scale,shape,skewness):
        return SkewtScore.mu_adj_score(y, mean, scale, shape, skewness)

    def draw_variable(self,loc,scale,shape,skewness,nsims):
        return loc + scale*dst.skewt.rvs(shape,skewness,nsims)

    def neg_loglik(self,beta):
        theta, Y, _ = self._model(beta)
        return -np.sum(dst.skewt.logpdf(x=Y,df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=self.link(theta),gamma=self.parameters.parameter_list[-3].prior.transform(beta[-3]),
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2])))

    def default_score_function(self,y,mean,scale,shape,skewness):
        return SkewtScore.mu_score(y, mean, scale, shape,skewness)

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
            t_params = self.transform_parameters()

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

            fitted_values = self.link(mu)+((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1]))

            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,fitted_values,label='GAS Filter',c='black')
            plt.legend(loc=2)   

            plt.subplot(2,1,2)
            plt.title("Filtered values for " + self.data_name)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, error_bars[0], error_bars[1], alpha=0.15,label='95% C.I. for Bootstrapped GAS')    

            plt.plot(date_index,fitted_values,label='GAS Filter',c='black')
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
            mean_values = self._mean_prediction(theta,Y,scores,h,t_params) + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1]))
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
            x = GASskewt(ar=self.ar,sc=self.sc,integ=self.integ,data=self.data_original[:-h+t])
            x.fit(printer=False)
            
            if t == 0:
                predictions = x.predict(1)
            else:
                predictions = pd.concat([predictions,x.predict(1)])
        
        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

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
            forecasted_values = mean_values[-h:]+((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1]))
            result = pd.DataFrame(forecasted_values)
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result