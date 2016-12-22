import numpy as np
import pandas as pd

from .covariances import acf
from .output import TablePrinter
from .inference import *
from .families import Normal, t, TruncatedNormal, Cauchy, Skewt, InverseGamma, Flat, InverseWishart, Laplace

class LatentVariables(object):
    """ Latent Variables Class

    Holds latent variable objects and contains method for latent variable manipulation. Latent variables are 
    referred to as z as shorthand. This is convention in much of the literature.
    """

    def __init__(self,model_name):
        self.model_name = model_name
        self.z_list = []
        self.z_indices = {}
        self.estimated = False
        self.estimation_method = None

    def __str__(self):
        z_row = []
        z_names = self.get_z_names()
        priors = self.get_z_priors()
        prior_names, prior_z_names = self.get_z_priors_names()
        vardist_names = self.get_z_approx_dist_names()
        transforms = self.get_z_transforms_names()

        fmt = [
            ('Index','z_index',8),      
            ('Latent Variable','z_name',25),
            ('Prior','z_prior',15),
            ('Prior Hyperparameters','z_hyper',25),
            ('V.I. Dist','z_vardist',10),
            ('Transform','z_transform',10)
            ]       

        for z in range(len(self.z_list)):
            z_row.append({'z_index': z, 'z_name': z_names[z],
                'z_prior': prior_names[z], 'z_hyper': prior_z_names[z],
                'z_vardist': vardist_names[z], 'z_transform': transforms[z]})
        return( TablePrinter(fmt, ul='=')(z_row) )

    def add_z(self, name, prior, q, index=True):
        """ Adds latent variable

        Parameters
        ----------
        name : str
            Name of the latent variable

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        q : Distribution object
            Which distribution to use for variational approximation

        index : boolean
            Whether to index the variable in the z_indices dictionary

        Returns
        ----------
        None (changes priors in LatentVariables object)
        """

        self.z_list.append(LatentVariable(name,len(self.z_list),prior,q))
        if index is True:
            self.z_indices[name] = {'start': len(self.z_list)-1, 'end': len(self.z_list)-1}

    def create(self, name, dim, prior, q):
        """ Creates multiple latent variables

        Parameters
        ----------
        name : str
            Name of the latent variable

        dim : list
            Dimension of the latent variable arrays

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        q : Distribution object
            Which distribution to use for variational approximation

        Returns
        ----------
        None (changes priors in LatentVariables object)
        """   

        def rec(dim, prev=[]):
           if len(dim) > 0:
               return [rec(dim[1:], prev + [i]) for i in range(dim[0])]
           else:
               return "(" + ",".join([str(j) for j in prev]) + ")" 

        indices = rec(dim)

        for f_dim in range(1, len(dim)):
            indices = sum(indices, [])

        if self.z_list is None:
            starting_index = 0
        else:
            starting_index = len(self.z_list)

        self.z_indices[name] = {'start': starting_index, 'end': starting_index+len(indices)-1, 'dim': len(dim)}

        for index in indices:
            self.add_z(name + " " + index, prior, q, index=False)

    def adjust_prior(self,index,prior):
        """ Adjusts priors for the latent variables

        Parameters
        ----------
        index : int or list[int]
            Which latent variable index/indices to be altered

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        Returns
        ----------
        None (changes priors in Parameters object)
        """

        if isinstance(index, list):
            for item in index:
                if item < 0 or item > (len(self.z_list)-1) or not isinstance(item, int):
                    raise ValueError("Oops - the latent variable index " + str(item) + " you have entered is invalid!")
                else:
                    self.z_list[item].prior = prior 
                    if hasattr(self.z_list[item].prior, 'mu0'):
                        self.z_list[item].start = self.z_list[item].prior.mu0    
                    elif hasattr(self.z_list[item].prior, 'loc0'):
                        self.z_list[item].start = self.z_list[item].prior.loc0
        else:
            if index < 0 or index > (len(self.z_list)-1) or not isinstance(index, int):
                raise ValueError("Oops - the latent variable index " + str(index) + " you have entered is invalid!")
            else:
                self.z_list[index].prior = prior
                if hasattr(self.z_list[index].prior, 'mu0'):
                    self.z_list[index].start = self.z_list[index].prior.mu0  
                elif hasattr(self.z_list[index].prior, 'loc0'):
                    self.z_list[index].start = self.z_list[index].prior.loc0  

    def get_z_names(self):
        names = []
        for z in self.z_list:
            names.append(z.name)
        return names

    def get_z_priors(self):
        priors = []
        for z in self.z_list:
            priors.append(z.prior)
        return priors

    def get_z_priors_names(self):
        priors = self.get_z_priors()
        prior_names = []
        prior_z_names = []
        for prior in priors:
            if isinstance(prior, Normal):
                prior_names.append('Normal')
                prior_z_names.append('mu0: ' + str(np.round(prior.mu0,4)) + ', sigma0: ' + str(np.round(prior.sigma0,4)))
            elif isinstance(prior, Laplace):
                prior_names.append('Laplace')
                prior_z_names.append('loc0: ' + str(np.round(prior.loc0,4)) + ', scale0: ' + str(np.round(prior.scale0,4)))
            elif isinstance(prior, InverseGamma):
                prior_names.append('Inverse Gamma')
                prior_z_names.append('alpha: ' + str(np.round(prior.alpha,4)) + ', beta: ' + str(np.round(prior.beta,4)))
            elif isinstance(prior, Flat):
                prior_names.append('Flat')
                prior_z_names.append('n/a (non-informative)')
            elif isinstance(prior, InverseWishart):
                prior_names.append('InverseWishart')
                prior_z_names.append('v: ' + str(np.round(prior.v,4)) + ' and scale matrix')
            elif isinstance(prior, t):
                prior_names.append('t')
                prior_z_names.append('loc0: ' + str(np.round(prior.loc0,4)) + ', scale0: ' + str(np.round(prior.scale0,4)) 
                    + ', df0: ' + str(np.round(prior.df0,4))) 
            elif isinstance(prior, Skewt):
                prior_names.append('Skewt')
                prior_z_names.append('loc0: ' + str(np.round(prior.loc0,4)) + ', scale0: ' + str(np.round(prior.scale0,4)) 
                    + ', df0: ' + str(np.round(prior.df0,4)) + ', gamma0: ' + str(np.round(prior.gamma0,4))) 
            elif isinstance(prior, Cauchy):
                prior_names.append('Cauchy')
                prior_z_names.append('loc0: ' + str(np.round(prior.loc0,4)) + ', scale0: ' + str(np.round(prior.scale0,4)))
            elif isinstance(prior, TruncatedNormal):
                prior_names.append('TruncatedNormal')
                prior_z_names.append('mu0: ' + str(np.round(prior.mu0,4)) + ', sigma0: ' + str(np.round(prior.sigma0,4)))                 
            else:
                raise ValueError("Prior distribution not detected!")
        return prior_names, prior_z_names

    def get_z_transforms(self):
        transforms = []
        for z in self.z_list:
            transforms.append(z.prior.transform)
        return transforms

    def get_z_transforms_names(self):
        transforms = []
        for z in self.z_list:
            transforms.append(z.prior.transform_name)
        return transforms

    def get_z_starting_values(self,transformed=False):
        transforms = self.get_z_transforms()

        values = np.zeros(len(self.z_list))
        for i in range(len(self.z_list)):
            if transformed is True:
                values[i] = transforms[i](self.z_list[i].start)
            else:
                values[i] = self.z_list[i].start
        return values

    def get_z_values(self, transformed=False):
        transforms = self.get_z_transforms()

        if self.estimated is True:
            values = np.zeros(len(self.z_list))
            for i in range(len(self.z_list)):
                if transformed is True:
                    values[i] = transforms[i](self.z_list[i].value)
                else:
                    values[i] = self.z_list[i].value
            return values
        else:
            return ValueError("No latent variables have been estimated yet")

    def get_z_approx_dist(self):
        dists = []
        for z in self.z_list:
            dists.append(z.q)
        return dists

    def get_z_approx_dist_names(self):
        approx_dists = self.get_z_approx_dist()
        q_list = []

        for approx in approx_dists:
            if isinstance(approx, Normal):
                q_list.append('Normal')
            else:
                raise Exception("Approximate distribution not detected!")
        return q_list

    def set_z_values(self,values,method,std=None,sample=None):
        if len(values) != len(self.z_list):
            raise ValueError("Length of your list is not equal to number of latent variables")
        for no in range(len(self.z_list)):
            self.z_list[no].method = method
            self.z_list[no].value = values[no]
            if std is not None:
                self.z_list[no].std = std[no]
            if sample is not None:
                self.z_list[no].sample = sample[no]
            self.estimated = True

    def set_z_starting_values(self,values):
        if values.shape[0] != len(self.z_list):
            raise ValueError("Length of your array is not equal to number of latent variables")
        for no in range(len(self.z_list)):
            self.z_list[no].start = values[no]

    def plot_z(self,indices=None,figsize=(15,5),loc=1):
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        import seaborn as sns

        plt.figure(figsize=figsize) 
        for z in range(1,len(self.z_list)+1):
            if indices is not None and z-1 not in indices:
                continue
            else:
                if hasattr(self.z_list[z-1], 'sample'):
                    sns.distplot(self.z_list[z-1].prior.transform(self.z_list[z-1].sample), rug=False, hist=False,label=self.z_list[z-1].method + ' estimate of ' + self.z_list[z-1].name)

                elif hasattr(self.z_list[z-1], 'value') and hasattr(self.z_list[z-1], 'std'): 

                    if self.z_list[z-1].prior.transform_name is None:
                        x = np.linspace(self.z_list[z-1].value-self.z_list[z-1].std*3.5,self.z_list[z-1].value+self.z_list[z-1].std*3.5,100)
                        plt.plot(x,mlab.normpdf(x,self.z_list[z-1].value,self.z_list[z-1].std),label=self.z_list[z-1].method + ' estimate of ' + self.z_list[z-1].name)
                    else:
                        sims = self.z_list[z-1].prior.transform(np.random.normal(self.z_list[z-1].value,self.z_list[z-1].std,100000))
                        sns.distplot(sims, rug=False, hist=False,label=self.z_list[z-1].method + ' estimate of ' + self.z_list[z-1].name)


                else:
                    raise ValueError("No information on latent variable to plot!")        

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Latent Variable Plot')
        plt.legend(loc=1)
        plt.show()

    def trace_plot(self,figsize=(15,15)):
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        import seaborn as sns

        if hasattr(self.z_list[0], 'sample'):
            fig = plt.figure(figsize=figsize)
            
            palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
            (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), 
            (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), 
            (0.5058823529411764, 0.4470588235294118, 0.6980392156862745), 
            (0.8, 0.7254901960784313, 0.4549019607843137), 
            (0.39215686274509803, 0.7098039215686275, 0.803921568627451)] * len(self.z_list)
            
            for j in range(len(self.z_list)):
                chain = self.z_list[j].sample
                for k in range(4):
                    iteration = j*4 + k + 1
                    ax = fig.add_subplot(len(self.z_list),4,iteration)
                    if iteration in range(1,len(self.z_list)*4 + 1,4):
                        a = sns.distplot(self.z_list[j].prior.transform(chain), rug=False, hist=False,color=palette[j])
                        a.set_ylabel(self.z_list[j].name)
                        if iteration == 1:
                            a.set_title('Density Estimate')
                    elif iteration in range(2,len(self.z_list)*4 + 1,4):
                        a = plt.plot(self.z_list[j].prior.transform(chain),color=palette[j])
                        if iteration == 2:
                            plt.title('Trace Plot')
                    elif iteration in range(3,len(self.z_list)*4 + 1,4): 
                        plt.plot(np.cumsum(self.z_list[j].prior.transform(chain))/np.array(range(1,len(chain)+1)),color=palette[j])
                        if iteration == 3:
                            plt.title('Cumulative Average')                 
                    elif iteration in range(4,len(self.z_list)*4 + 1,4):
                        plt.bar(range(1,10),[acf(chain,lag) for lag in range(1,10)],color=palette[j])
                        if iteration == 4:
                            plt.title('ACF Plot')                       
            sns.plt.show()  
        else:
            raise ValueError("No samples to plot!")


class LatentVariable(object):
    """ LatentVariable Class

    Parameters
    ----------
    name : str
        Name of the latent variable

    index : int
        Index of the latent variable

    prior : Prior object
        The prior for the latent variable, e.g. Normal(0,1)

    q : fam.[dist] object
        The variational distribution for the latent variable, e.g. Normal(0,1)
    """

    def __init__(self,name,index,prior,q):
        self.name = name
        self.index = 0
        self.prior = prior
        self.transform = self.prior.transform
        self.start = 0.0
        self.q = q

    def plot_z(self,figsize=(15,5)):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if hasattr(self, 'sample'):
            sns.distplot(self.prior.transform(self.sample), rug=False, hist=False,label=self.method + ' estimate of ' + self.name)

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
            raise ValueError("No information on latent variable to plot!")