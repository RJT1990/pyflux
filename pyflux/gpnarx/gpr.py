import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices, dmatrix, demo_data

from .. import arma
from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .kernels import *
from .gp import *

class GPR(GP):
    """ Inherits time series methods from TSM class.

    **** GAUSSIAN PROCESS REGRESSION ****

    Parameters
    ----------
    formula : string
        Patsy regression

    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    kernel_type : str
        One of SE (SquaredExponential), OU (Ornstein-Uhlenbeck), RQ
        (RationalQuadratic), Periodic, ARD. Defines kernel choice for GP-NARX.
    """

    def __init__(self,formula,data,kernel_type='SE'):

        # Initialize TSM object
        super(GP,self).__init__('GP')

        # Parameters
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.max_lag = 0
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
        self.X = np.array([self.X])[0].T
        self.index = data.index
        self.X_original = self.X.copy()

        if kernel_type == 'ARD':
            self.param_no += 2 
        elif kernel_type == 'RQ':
            self.param_no = 3 + 1
        else:
            self.param_no = 3

        self.model_name = 'GPR'
        self.kernel_type = kernel_type

        # Apply normalization
        self.data = self.y.copy()
        self._norm_mean = np.mean(self.data)
        self._norm_std = np.std(self.data)  
        self.data = (self.data - self._norm_mean) / self._norm_std

        for regressor in range(self.X.shape[0]):
            self.X[regressor] = (self.X[regressor] - np.mean(self.X[regressor])) / np.std(self.X[regressor])

        self._create_parameters()

    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        self.parameters.add_parameter('Noise Sigma^2',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        if self.kernel_type == 'ARD':
            self.kernel = ARD(self.X,np.ones(len(self.X_names)),1)

            for reg in range(self.X.shape[0]):
                self.parameters.add_parameter('l-' + self.X_names[reg],ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        else:
            self.parameters.add_parameter('l',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

            if self.kernel_type == 'SE':
                self.kernel = SquaredExponential(self.X,1,1)
            elif self.kernel_type == 'OU':
                self.kernel = OrnsteinUhlenbeck(self.X,1,1)
            elif self.kernel_type == 'Periodic':
                self.kernel = Periodic(self.X,1,1)
            elif self.kernel_type == 'RQ':
                self.parameters.add_parameter('alpha',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
                self.kernel = RationalQuadratic(self.X,1,1,1)

        self.parameters.add_parameter('tau',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

    def _start_params(self,beta):
        """ Transforms parameters for use in kernels

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        None (changes data in self.kernel)
        """

        if self.kernel_type == 'ARD':
            for reg in range(self.X.shape[0]):
                self.kernel.l[reg] = self.parameters.parameter_list[1+reg].prior.transform(beta[1+reg])
            self.kernel.tau = self.parameters.parameter_list[-1].prior.transform(beta[-1])  
        elif self.kernel_type == 'RQ':
            self.kernel.l = self.parameters.parameter_list[1].prior.transform(beta[1])
            self.kernel.a = self.parameters.parameter_list[2].prior.transform(beta[2])          
            self.kernel.tau = self.parameters.parameter_list[3].prior.transform(beta[3])    
        else:
            self.kernel.l = self.parameters.parameter_list[1].prior.transform(beta[1])
            self.kernel.tau = self.parameters.parameter_list[2].prior.transform(beta[2])            

    def plot_oned(self,figsize=(15,10),intervals=True):
        if self.parameters.estimated is True and len(self.X_names) == 1:
            X_sort = self.X.copy()
            for parameter in range(X_sort.shape[0]):
                X_sort[parameter] = np.sort(self.X[parameter])

            Kstar = self.kernel.Kstar(np.transpose(np.array(X_sort)))
            L = self._L(self.parameters.get_parameter_values())
            output =  np.dot(np.transpose(Kstar),self._alpha(L))

            posterior = multivariate_normal(output,self.variance_values(self.parameters.get_parameter_values()),allow_singular=True)
            simulations = 500
            sim_vector = np.zeros([simulations,len(output)])

            for i in range(simulations):
                sim_vector[i] = posterior.rvs()

            error_bars = []
            for pre in range(5,100,5):
                error_bars.append([(np.percentile(i,pre)*self._norm_std + self._norm_mean) for i in sim_vector.transpose()] 
                    - (output*self._norm_std + self._norm_mean))

            plt.figure(figsize=figsize)
            plt.plot(X_sort[0]*np.std(self.X_original[0]) + np.mean(self.X[0]),output*self._norm_std + self._norm_mean)

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(X_sort[0]*np.std(self.X_original[0]) + np.mean(self.X[0]), (output*self._norm_std + self._norm_mean)-pre, 
                        (output*self._norm_std + self._norm_mean)+pre,alpha=alpha[count])       

            plt.title(self.y_name + "=" + "f(" + self.X_names[0] + ")")
            plt.xlabel(self.X_names[0])
            plt.ylabel(self.y_name)
            plt.show()
        elif self.parameters.estimated is True and len(self.X_names) != 1:
            raise ValueError("Not a 1-D problem")
        else:
            raise ValueError("No parameters have been estimated yet")

    def plot_fit(self,intervals=True,**kwargs):
        """ Plots the fit of the Gaussian process model to the data

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        intervals : Boolean
            Whether to plot uncertainty intervals or not

        Returns
        ----------
        None (plots the fit of the function)
        """

        figsize = kwargs.get('figsize',(10,7))

        date_index = self.index
        expectation = self.expected_values(self.parameters.get_parameter_values())
        posterior = multivariate_normal(expectation,self.variance_values(self.parameters.get_parameter_values()),allow_singular=True)
        simulations = 500
        sim_vector = np.zeros([simulations,len(expectation)])

        for i in range(simulations):
            sim_vector[i] = posterior.rvs()

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append([(np.percentile(i,pre)*self._norm_std + self._norm_mean) for i in sim_vector.transpose()] 
                - (expectation*self._norm_std + self._norm_mean))

        plt.figure(figsize=figsize) 

        plt.subplot(2, 2, 1)
        plt.title(self.data_name + " Raw")  
        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k')

        plt.subplot(2, 2, 2)
        plt.title(self.data_name + " Raw and Expected") 
        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k',alpha=0.2)
        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.subplot(2, 2, 3)
        plt.title(self.data_name + " Raw and Expected (with intervals)")    

        if intervals == True:
            alpha =[0.15*i/float(100) for i in range(50,12,-2)]
            for count, pre in enumerate(error_bars):
                plt.fill_between(date_index, (expectation*self._norm_std + self._norm_mean)-pre, 
                    (expectation*self._norm_std + self._norm_mean)+pre,alpha=alpha[count])      

        plt.plot(date_index,self.data*self._norm_std + self._norm_mean,'k',alpha=0.2)
        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.subplot(2, 2, 4)

        plt.title("Expected " + self.data_name + " (with intervals)")   

        if intervals == True:
            alpha =[0.15*i/float(100) for i in range(50,12,-2)]
            for count, pre in enumerate(error_bars):
                plt.fill_between(date_index, (expectation*self._norm_std + self._norm_mean)-pre, 
                    (expectation*self._norm_std + self._norm_mean)+pre,alpha=alpha[count])      

        plt.plot(date_index,self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean,'b')

        plt.show()