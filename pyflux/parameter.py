import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

from .covariances import acf
from .output import TablePrinter
from .inference import *
from .distributions import *

class Parameters(object):
    """ Parameters Class

    Holds parameter objects and contains method for parameter manipulation
    """

    def __init__(self,model_name):
        self.model_name = model_name
        self.parameter_list = []
        self.estimated = False

    def __str__(self):
        param_row = []
        parameter_names = self.get_parameter_names()
        priors = self.get_parameter_priors()
        prior_names, prior_params_names = self.get_parameter_priors_names()
        vardist_names = self.get_parameter_approx_dist_names()
        transforms = self.get_parameter_transforms_names()

        fmt = [
            ('Index','param_index',6),      
            ('Parameter','param_name',25),
            ('Prior','param_prior',10),
            ('Hyperparameters','param_hyper',25),
            ('V.I. Dist','param_vardist',10),
            ('Transform','param_transform',10)
            ]       

        for param in range(len(self.parameter_list)):
            param_row.append({'param_index': param, 'param_name': parameter_names[param],
                'param_prior': prior_names[param], 'param_hyper': prior_params_names[param],
                'param_vardist': vardist_names[param], 'param_transform': transforms[param]})
        return( TablePrinter(fmt, ul='=')(param_row) )

    def add_parameter(self,name,prior,q):
        """ Adds parameter

        Parameters
        ----------
        name : str
            Name of the parameter

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        q : Distribution object
            Which distribution to use for variational approximation

        Returns
        ----------
        None (changes priors in Parameters object)
        """

        self.parameter_list.append(Parameter(name,len(self.parameter_list),prior,q))

    def adjust_prior(self,index,prior):
        """ Adjusts priors for the parameters

        Parameters
        ----------
        index : int or list[int]
            Which parameter index/indices to be altered

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        Returns
        ----------
        None (changes priors in Parameters object)
        """

        if isinstance(index, list):
            for item in index:
                if item < 0 or item > (len(self.parameter_list)-1) or not isinstance(item, int):
                    raise ValueError("Oops - the parameter index " + str(item) + " you have entered is invalid!")
                else:
                    self.parameter_list[item].prior = prior             
        else:
            if index < 0 or index > (len(self.parameter_list)-1) or not isinstance(index, int):
                raise ValueError("Oops - the parameter index " + str(index) + " you have entered is invalid!")
            else:
                self.parameter_list[index].prior = prior

    def get_parameter_names(self):
        names = []
        for parameter in self.parameter_list:
            names.append(parameter.name)
        return names

    def get_parameter_priors(self):
        priors = []
        for parameter in self.parameter_list:
            priors.append(parameter.prior)
        return priors

    def get_parameter_priors_names(self):
        priors = self.get_parameter_priors()
        prior_names = []
        prior_params_names = []
        for prior in priors:
            if isinstance(prior, Normal):
                prior_names.append('Normal')
                prior_params_names.append('mu0: ' + str(prior.mu0) + ', sigma0: ' + str(prior.sigma0))
            elif isinstance(prior, InverseGamma):
                prior_names.append('Inverse Gamma')
                prior_params_names.append('alpha: ' + str(prior.alpha) + ', beta: ' + str(prior.beta))
            elif isinstance(prior, Uniform):
                prior_names.append('Uniform')
                prior_params_names.append('n/a (non-informative)')
            else:
                raise ValueError("Prior distribution not detected!")
        return prior_names, prior_params_names

    def get_parameter_transforms(self):
        transforms = []
        for parameter in self.parameter_list:
            transforms.append(parameter.prior.transform)
        return transforms

    def get_parameter_transforms_names(self):
        transforms = []
        for parameter in self.parameter_list:
            transforms.append(parameter.prior.transform_name)
        return transforms

    def get_parameter_starting_values(self,transformed=False):
        transforms = self.get_parameter_transforms()

        values = np.zeros(len(self.parameter_list))
        for i in range(len(self.parameter_list)):
            if transformed is True:
                values[i] = transforms[i](self.parameter_list[i].start)
            else:
                values[i] = self.parameter_list[i].start
        return values

    def get_parameter_values(self,transformed=False):
        transforms = self.get_parameter_transforms()

        if self.estimated is True:
            values = np.zeros(len(self.parameter_list))
            for i in range(len(self.parameter_list)):
                if transformed is True:
                    values[i] = transforms[i](self.parameter_list[i].value)
                else:
                    values[i] = self.parameter_list[i].value
            return values
        else:
            return ValueError("No parameters have been estimated yet")

    def get_parameter_approx_dist(self):
        dists = []
        for parameter in self.parameter_list:
            dists.append(parameter.q)
        return dists

    def get_parameter_approx_dist_names(self):
        approx_dists = self.get_parameter_approx_dist()
        q_list = []

        for approx in approx_dists:
            if isinstance(approx, q_Normal):
                q_list.append('Normal')
            elif isinstance(approx, q_InverseGamma):
                q_list.append('Inverse Gamma')
            elif isinstance(approx, q_Uniform):
                q_list.append('Uniform')
            else:
                raise Exception("Approximate distribution not detected!")
        return q_list

    def set_parameter_values(self,values,method,std=None,sample=None):
        if len(values) != len(self.parameter_list):
            raise ValueError("Length of your list is not equal to length of parameters")
        for no in range(len(self.parameter_list)):
            self.parameter_list[no].method = method
            self.parameter_list[no].value = values[no]
            if std is not None:
                self.parameter_list[no].std = std[no]
            if sample is not None:
                self.parameter_list[no].sample = sample[no]
            self.estimated = True

    def set_parameter_starting_values(self,values):
        if values.shape[0] != len(self.parameter_list):
            raise ValueError("Length of your array is not equal to number of parameters")
        for no in range(len(self.parameter_list)):
            self.parameter_list[no].start = values[no]

    def plot_parameters(self,indices=None,figsize=(15,5)):
        plt.figure(figsize=figsize) 
        for parm in range(1,len(self.parameter_list)+1):
            if indices is not None and parm-1 not in indices:
                continue
            else:
                if hasattr(self.parameter_list[parm-1], 'sample'):
                    sns.distplot(self.parameter_list[parm-1].sample, rug=False, hist=False,label=self.parameter_list[parm-1].method + ' estimate of ' + self.parameter_list[parm-1].name)

                elif hasattr(self.parameter_list[parm-1], 'value') and hasattr(self.parameter_list[parm-1], 'std'): 

                    if self.parameter_list[parm-1].prior.transform_name is None:
                        x = np.linspace(self.parameter_list[parm-1].value-self.parameter_list[parm-1].std*3.5,self.parameter_list[parm-1].value+self.parameter_list[parm-1].std*3.5,100)
                        plt.plot(x,mlab.normpdf(x,self.parameter_list[parm-1].value,self.parameter_list[parm-1].std),label=self.parameter_list[parm-1].method + ' estimate of ' + self.parameter_list[parm-1].name)
                    else:
                        sims = self.parameter_list[parm-1].prior.transform(np.random.normal(self.parameter_list[parm-1].value,self.parameter_list[parm-1].std,100000))
                        sns.distplot(sims, rug=False, hist=False,label=self.parameter_list[parm-1].method + ' estimate of ' + self.parameter_list[parm-1].name)


                else:
                    raise ValueError("No information on parameter to plot!")        

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Parameter Plot')
        plt.legend()
        plt.show()

    def trace_plot(self,figsize=(15,15)):
        if hasattr(self.parameter_list[0], 'sample'):
            fig = plt.figure(figsize=figsize)

            for j in range(len(self.parameter_list)):
                chain = self.parameter_list[j].sample
                for k in range(4):
                    iteration = j*4 + k + 1
                    ax = fig.add_subplot(len(self.parameter_list),4,iteration)

                    if iteration in range(1,len(self.parameter_list)*4 + 1,4):
                        a = sns.distplot(chain, rug=False, hist=False)
                        a.set_ylabel(self.parameter_list[j].name)
                        if iteration == 1:
                            a.set_title('Density Estimate')
                    elif iteration in range(2,len(self.parameter_list)*4 + 1,4):
                        a = plt.plot(chain)
                        if iteration == 2:
                            plt.title('Trace Plot')
                    elif iteration in range(3,len(self.parameter_list)*4 + 1,4): 
                        plt.plot(np.cumsum(chain)/np.array(range(1,len(chain)+1)))
                        if iteration == 3:
                            plt.title('Cumulative Average')                 
                    elif iteration in range(4,len(self.parameter_list)*4 + 1,4):
                        plt.bar(range(1,10),[acf(chain,lag) for lag in range(1,10)])
                        if iteration == 4:
                            plt.title('ACF Plot')                       
            sns.plt.show()  
        else:
            raise ValueError("No samples to plot!")


class Parameter(object):
    """ Parameter Class

    Parameters
    ----------
    name : str
        Name of the parameter

    index : int
        Index of the parameter

    prior : Prior object
        The prior for the parameter, e.g. Normal(0,1)

    q : dst.q_[dist] object
        The variational distribution for the parameter, e.g. q_Normal(0,1)
    """

    def __init__(self,name,index,prior,q):
        self.name = name
        self.index = 0
        self.prior = prior
        self.transform = self.prior.transform
        self.start = 0.0
        self.q = q

    def plot_parameter(self,figsize=(15,5)):
        if hasattr(self, 'sample'):
            sns.distplot(self.sample, rug=False, hist=False,label=self.method + ' estimate of ' + self.name)

        elif hasattr(self, 'value') and hasattr(self, 'std'):
            x = np.linspace(self.value-self.std*3.5,self.value+self.std*3.5,100)
            plt.figure(figsize=figsize)
            if self.prior.transform_name is None:
                plt.plot(x,mlab.normpdf(x,self.value,self.std),label=self.method + ' estimate of ' + self.name)
            else:
                sims = self.prior.transform(np.random.normal(self.value,self.std,100000))
                sns.distplot(sims, rug=False, hist=False,label=self.method + ' estimate of ' + self.name)
            plt.xlabel('Value')
            plt.legend()
            plt.show()

        else:
            raise ValueError("No information on parameter to plot!")