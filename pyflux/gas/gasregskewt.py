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
from .gasreg import *

class GASRegskewt(GASReg):
    """ Inherits time series methods from TSM class.

    **** GAS skewt REGRESSION MODELS ****

    Parameters
    ----------

    formula : string
        patsy string describing the regression

    data : pd.DataFrame or np.array
        Field to specify the data that will be used
    """

    def __init__(self,formula,data):

        # Initialize TSM object     
        super(GASRegskewt,self).__init__(formula=formula,data=data)

        self.model_name = "skewt-distributed GAS Regression"
        self.dist = 'skewt'
        self.link = np.array
        self.scale = True
        self.shape = True
        self.skewness = True

        for parm in range(len(self.parameters.parameter_list)):
            self.parameters.parameter_list[parm].start = -9.0

        self.parameters.add_parameter('skewness',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('t Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        self.parameters.parameter_list[-2].start = 0.0
        self.parameters.parameter_list[-1].start = 2.0

        self.param_no += 3

    def score_function(self,X,y,mean,scale,shape,gamma):
        if (y-mean)>=0:
            return ((shape+1)/shape)*((y-mean)*X)/(power(gamma*scale,2) + (power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*((y-mean)*X)/(power(scale,2) + (power(gamma*(y-mean),2)/shape))

    def draw_variable(self,loc,scale,shape,skewness,nsims):
        return loc + scale*dst.skewt.rvs(shape,skewness,nsims)

    def neg_loglik(self,beta):
        theta, Y, scores,_ = self._model(beta)
        return -np.sum(dst.skewt.logpdf(x=Y,df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=self.link(theta),gamma=self.parameters.parameter_list[-3].prior.transform(beta[-3]),
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2])))

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
            t_params = self.transform_parameters()

            plt.figure(figsize=figsize) 
            
            plt.subplot(len(self.X_names)+1, 1, 1)
            plt.title(self.y_name + " Filtered")
            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,self.link(mu)+((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1])),label='GAS Filter',c='black')
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

            plot_values = np.append(self.y,self.link(theta_pred)+((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1])))
            plot_values = plot_values[-h-past_values:] 
            forecasted_values = np.append(self.y[-1],self.link(theta_pred)) +((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1]))
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
            x = GASRegskewt(formula=self.formula,data=self.data_original[:(-h+t)])
            x.fit(printer=False)
            if t == 0:
                predictions = x.predict(1,oos_data=data2)
            else:
                predictions = pd.concat([predictions,x.predict(h=1,oos_data=data2)])
        
        predictions.rename(columns={0:self.y_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

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
            t_params = self.transform_parameters()

            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            X_pred = X_oos[:h]
            date_index = self.shift_dates(h)
            _, _, _, coefficients = self._model(self.parameters.get_parameter_values()) 
            coefficients_star = coefficients.T[-1]
            theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0] +((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*SkewtScore.tv_variate_exp(t_params[-1]))

            result = pd.DataFrame(self.link(theta_pred))
            result.rename(columns={0:self.y_name}, inplace=True)
            result.index = date_index[-h:]

            return result