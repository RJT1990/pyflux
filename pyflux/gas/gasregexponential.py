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

class GASRegExponential(GASReg):
    """ Inherits time series methods from TSM class.

    **** GAS EXPONENTIAL REGRESSION MODELS ****

    Parameters
    ----------

    formula : string
        patsy string describing the regression

    data : pd.DataFrame or np.array
        Field to specify the data that will be used
    """

    def __init__(self,formula,data):

        # Initialize TSM object     
        super(GASRegExponential,self).__init__(formula=formula,data=data)

        self.model_name = "Exponential-distributed GAS Regression"
        self.dist = 'Exponential'
        self.link = np.exp
        self.scale = False
        self.shape = False

        for parm in range(len(self.parameters.parameter_list)):
            self.parameters.parameter_list[parm].start = -9.0

    def score_function(self,X,y,mean,scale,shape,skewness):
        return X*(1.0 - mean*y)

    def draw_variable(self,loc,scale,shape,skewness,nsims):
        return np.random.exponential(1/loc, nsims)

    def neg_loglik(self,beta):
        theta, Y, scores,_ = self._model(beta)
        return -np.sum(ss.expon.logpdf(x=Y,scale=1/self.link(theta)))

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
            x = GASRegPoisson(formula=self.formula,data=self.data_original[:(-h+t)])
            x.fit(printer=False)
            if t == 0:
                predictions = x.predict(1,oos_data=data2)
            else:
                predictions = pd.concat([predictions,x.predict(h=1,oos_data=data2)])
        
        predictions.rename(columns={0:self.y_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions