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

class GASt(GAS):
    """ Inherits GAS methods from GAS class (and time series methods from TSM class).

    **** t GENERALIZED AUTOREGRESSIVE SCORE (GAS) MODELS ****

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
        super(GASt,self).__init__(data=data,ar=ar,sc=sc,integ=integ,
            target=target,gradient_only=gradient_only)

        self.model_name = "t GAS(" + str(self.ar) + "," + str(self.integ) + "," + str(self.sc) + ") REGRESSION"     
        self.dist = 't'
        self.link = np.array
        self.scale = True
        self.shape = True
        self.param_no += 2
        self.parameters.add_parameter('Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        self.parameters.parameter_list[0].start = np.mean(self.data)
        self.parameters.parameter_list[-1].start = 2.0

        if gradient_only is False:
            self.score_function = self.adj_score_function
        else:
            self.score_function = self.default_score_function

    def adj_score_function(self,y,mean,scale,shape):
        return tScore.mu_adj_score(y, mean, scale, shape)

    def draw_variable(self,loc,scale,shape,nsims):
        return loc + scale*np.random.standard_t(shape,nsims)

    def neg_loglik(self,beta):
        theta, Y, _ = self._model(beta)
        return -np.sum(ss.t.logpdf(x=Y,df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=self.link(theta),
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2])))

    def default_score_function(self,y,mean,scale,shape):
        return tScore.mu_score(y, mean, scale, shape)

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
            x = GASt(ar=self.ar,sc=self.sc,integ=self.integ,data=self.data_original[:-h+t])
            x.fit(printer=False)
            
            if t == 0:
                predictions = x.predict(1)
            else:
                predictions = pd.concat([predictions,x.predict(1)])
        
        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions