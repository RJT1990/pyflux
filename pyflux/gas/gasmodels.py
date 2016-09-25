import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .. import inference as ifr
from .. import distributions as dst

# Import Cythonized recursions - spaced out for readability (at expense of conciseness!)
from .gas_recursions import gas_recursion_poisson_orderone, gas_recursion_poisson_ordertwo
from .gas_recursions import gas_recursion_normal_orderone, gas_recursion_normal_ordertwo
from .gas_recursions import gas_recursion_exponential_orderone, gas_recursion_exponential_ordertwo
from .gas_recursions import gas_recursion_laplace_orderone, gas_recursion_laplace_ordertwo
from .gas_recursions import gas_recursion_t_orderone, gas_recursion_t_ordertwo
from .gas_recursions import gas_recursion_skewt_orderone, gas_recursion_skewt_ordertwo

from .gas_recursions import gasx_recursion_poisson_orderone, gasx_recursion_poisson_ordertwo
from .gas_recursions import gasx_recursion_normal_orderone, gasx_recursion_normal_ordertwo
from .gas_recursions import gasx_recursion_exponential_orderone, gasx_recursion_exponential_ordertwo
from .gas_recursions import gasx_recursion_laplace_orderone, gasx_recursion_laplace_ordertwo
from .gas_recursions import gasx_recursion_t_orderone, gasx_recursion_t_ordertwo
from .gas_recursions import gasx_recursion_skewt_orderone, gasx_recursion_skewt_ordertwo

from .gas_recursions import gas_llev_recursion_poisson_orderone, gas_llev_recursion_poisson_ordertwo
from .gas_recursions import gas_llev_recursion_normal_orderone, gas_llev_recursion_normal_ordertwo
from .gas_recursions import gas_llev_recursion_exponential_orderone, gas_llev_recursion_exponential_ordertwo
from .gas_recursions import gas_llev_recursion_laplace_orderone, gas_llev_recursion_laplace_ordertwo
from .gas_recursions import gas_llev_recursion_t_orderone, gas_llev_recursion_t_ordertwo
from .gas_recursions import gas_llev_recursion_skewt_orderone, gas_llev_recursion_skewt_ordertwo

from .gas_recursions import gas_llt_recursion_poisson_orderone, gas_llt_recursion_poisson_ordertwo
from .gas_recursions import gas_llt_recursion_normal_orderone, gas_llt_recursion_normal_ordertwo
from .gas_recursions import gas_llt_recursion_exponential_orderone, gas_llt_recursion_exponential_ordertwo
from .gas_recursions import gas_llt_recursion_laplace_orderone, gas_llt_recursion_laplace_ordertwo
from .gas_recursions import gas_llt_recursion_t_orderone, gas_llt_recursion_t_ordertwo
from .gas_recursions import gas_llt_recursion_skewt_orderone, gas_llt_recursion_skewt_ordertwo

from .gas_recursions import gas_reg_recursion_poisson_orderone, gas_reg_recursion_poisson_ordertwo
from .gas_recursions import gas_reg_recursion_normal_orderone, gas_reg_recursion_normal_ordertwo
from .gas_recursions import gas_reg_recursion_exponential_orderone, gas_reg_recursion_exponential_ordertwo
from .gas_recursions import gas_reg_recursion_laplace_orderone, gas_reg_recursion_laplace_ordertwo
from .gas_recursions import gas_reg_recursion_t_orderone, gas_reg_recursion_t_ordertwo
from .gas_recursions import gas_reg_recursion_skewt_orderone, gas_reg_recursion_skewt_ordertwo

def exponential_link(x):
    """ Creates exponential link function for the exponential distribution

    Returns
    ----------
    - link transformation for the exponential distribution
    """
    return 1.0/np.exp(x)

class GASDistribution(object):
    """ GAS Distribution Class (parent class)

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=False):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

class GASExponential(GASDistribution):
    """ GAS Exponential Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "Exponential GAS"
        link = exponential_link
        scale = False
        shape = False
        skewness = False
        mean_transform = np.log
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from Exponential distribution        

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return np.random.exponential(1.0/loc, nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS Exponential Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Exponential distribution

        scale : float
            scale parameter for the Exponential distribution

        shape : float
            tail thickness parameter for the Exponential distribution

        skewness : float
            skewness parameter for the Exponential distribution

        Returns
        ----------
        - Score of the Exponential family
        """
        return 1 - (mean*y)

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Exponential distribution

        scale : float
            scale parameter for the Exponential distribution

        shape : float
            tail thickness parameter for the Exponential distribution

        skewness : float
            skewness parameter for the Exponential distribution

        Returns
        ----------
        - Negative loglikelihood of the Exponential family
        """
        return -np.sum(ss.expon.logpdf(x=y, scale=1/mean))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS Exponential Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Exponential distribution

        scale : float
            scale parameter for the Exponential distribution

        shape : float
            tail thickness parameter for the Exponential distribution

        skewness : float
            skewness parameter for the Exponential distribution

        Returns
        ----------
        - Adjusted score of the Exponential family
        """
        return 1 - (mean*y)

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS Exponential Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Exponential distribution

        scale : float
            scale parameter for the Exponential distribution

        shape : float
            tail thickness parameter for the Exponential distribution

        skewness : float
            skewness parameter for the Exponential distribution

        Returns
        ----------
        - Score of the Exponential family
        """
        return X*(1.0 - mean*y)

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS Exponential Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Exponential model - gradient only
        """
        return gas_recursion_exponential_orderone

    @staticmethod
    def newton_recursion():
        """ GAS Exponential Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Exponential model - adjusted score
        """
        return gas_recursion_exponential_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX Exponential Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX Exponential model - gradient only
        """
        return gasx_recursion_exponential_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX Exponential Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX Exponential model - adjusted score
        """
        return gasx_recursion_exponential_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level Exponential Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level Exponential model - gradient only
        """
        return gas_llev_recursion_exponential_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level Exponential Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level Exponential model - adjusted score
        """
        return gas_llev_recursion_exponential_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend Exponential Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Exponential model - gradient only
        """
        return gas_llt_recursion_exponential_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend Exponential Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Exponential model - adjusted score
        """
        return gas_llt_recursion_exponential_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression Exponential Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Exponential model - gradient only
        """
        return gas_reg_recursion_exponential_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression Exponential Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Exponential model - adjusted score
        """
        return gas_reg_recursion_exponential_ordertwo

class GASLaplace(GASDistribution):
    """ GAS Laplace Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "Laplace GAS"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Laplace Scale', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 2.0])
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from this distribution

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return np.random.laplace(loc, scale, nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS Laplace Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Laplace distribution

        scale : float
            scale parameter for the Laplace distribution

        shape : float
            tail thickness parameter for the Laplace distribution

        skewness : float
            skewness parameter for the Laplace distribution

        Returns
        ----------
        - Score of the Laplace family
        """
        return (y-mean)/float(scale*np.abs(y-mean))

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Laplace distribution

        scale : float
            scale parameter for the Laplace distribution

        shape : float
            tail thickness parameter for the Laplace distribution

        skewness : float
            skewness parameter for the Laplace distribution

        Returns
        ----------
        - Negative loglikelihood of the Laplace family
        """
        return -np.sum(ss.laplace.logpdf(y, loc=mean, scale=scale))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS Laplace Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Laplace distribution

        scale : float
            scale parameter for the Laplace distribution

        shape : float
            tail thickness parameter for the Laplace distribution

        skewness : float
            skewness parameter for the Laplace distribution

        Returns
        ----------
        - Adjusted score of the Laplace family
        """
        return ((y-mean)/float(scale*np.abs(y-mean))) / (-(np.power(y-mean,2) - np.power(np.abs(mean-y),2))/(scale*np.power(np.abs(mean-y),3)))

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS Laplace Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Laplace distribution

        scale : float
            scale parameter for the Laplace distribution

        shape : float
            tail thickness parameter for the Laplace distribution

        skewness : float
            skewness parameter for the Laplace distribution

        Returns
        ----------
        - Score of the Laplace family
        """
        return X*(y-mean)/(scale*np.abs(y-mean))

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS Laplace Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Laplace model - gradient only
        """
        return gas_recursion_laplace_orderone

    @staticmethod
    def newton_recursion():
        """ GAS Laplace Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Laplace model - adjusted score
        """
        return gas_recursion_laplace_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX Laplace Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX Laplace model - gradient only
        """
        return gasx_recursion_laplace_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX Laplace Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX Laplace model - adjusted score
        """
        return gasx_recursion_laplace_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level Laplace Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level Laplace model - gradient only
        """
        return gas_llev_recursion_laplace_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level Laplace Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level Laplace model - adjusted score
        """
        return gas_llev_recursion_laplace_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend Laplace Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Laplace model - gradient only
        """
        return gas_llt_recursion_laplace_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend Laplace Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Laplace model - adjusted score
        """
        return gas_llt_recursion_laplace_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression Laplace Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Laplace model - gradient only
        """
        return gas_reg_recursion_laplace_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression Laplace Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Laplace model - adjusted score
        """
        return gas_reg_recursion_laplace_ordertwo


class GASNormal(GASDistribution):
    """ GAS Normal Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=False):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "Normal GAS"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Normal Scale', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 0.0])
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from this distribution

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return np.random.normal(loc, scale, nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS Normal Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Normal distribution

        scale : float
            scale parameter for the Normal distribution

        shape : float
            tail thickness parameter for the Normal distribution

        skewness : float
            skewness parameter for the Normal distribution

        Returns
        ----------
        - Score of the Normal family
        """
        return (y-mean)/np.power(scale,2)

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Normal distribution

        scale : float
            scale parameter for the Normal distribution

        shape : float
            tail thickness parameter for the Normal distribution

        skewness : float
            skewness parameter for the Normal distribution

        Returns
        ----------
        - Negative loglikelihood of the Normal family
        """
        return -np.sum(ss.norm.logpdf(y, loc=mean, scale=scale))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS Normal Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Normal distribution

        scale : float
            scale parameter for the Normal distribution

        shape : float
            tail thickness parameter for the Normal distribution

        skewness : float
            skewness parameter for the Normal distribution

        Returns
        ----------
        - Adjusted score of the Normal family
        """
        return y-mean

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS Normal Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Normal distribution

        scale : float
            scale parameter for the Normal distribution

        shape : float
            tail thickness parameter for the Normal distribution

        skewness : float
            skewness parameter for the Normal distribution

        Returns
        ----------
        - Score of the Normal family
        """
        return X*(y-mean)

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS Normal Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Normal model - gradient only
        """
        return gas_recursion_normal_orderone

    @staticmethod
    def newton_recursion():
        """ GAS Normal Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Normal model - adjusted score
        """
        return gas_recursion_normal_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX Normal Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX Normal model - gradient only
        """
        return gasx_recursion_normal_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX Normal Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX Normal model - adjusted score
        """
        return gasx_recursion_normal_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level Normal Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level Normal model - gradient only
        """
        return gas_llev_recursion_normal_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level Normal Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level Normal model - adjusted score
        """
        return gas_llev_recursion_normal_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend Normal Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Normal model - gradient only
        """
        return gas_llt_recursion_normal_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend Normal Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Normal model - adjusted score
        """
        return gas_llt_recursion_normal_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression Normal Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Normal model - gradient only
        """
        return gas_reg_recursion_normal_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression Normal Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Normal model - adjusted score
        """
        return gas_reg_recursion_normal_ordertwo


class GASPoisson(GASDistribution):
    """ GAS Poisson Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=False):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "Poisson GAS"
        link = np.exp
        scale = False
        shape = False
        skewness = False
        mean_transform = np.log
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from Poisson distribution

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return np.random.poisson(loc, nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS Poisson Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Poisson distribution

        scale : float
            scale parameter for the Poisson distribution

        shape : float
            tail thickness parameter for the Poisson distribution

        skewness : float
            skewness parameter for the Poisson distribution

        Returns
        ----------
        - Score of the Poisson family
        """
        return y-mean

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Poisson distribution

        scale : float
            scale parameter for the Poisson distribution

        shape : float
            tail thickness parameter for the Poisson distribution

        skewness : float
            skewness parameter for the Poisson distribution

        Returns
        ----------
        - Negative loglikelihood of the Poisson family
        """
        return -np.sum(ss.poisson.logpmf(y, mean))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS Poisson Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Poisson distribution

        scale : float
            scale parameter for the Poisson distribution

        shape : float
            tail thickness parameter for the Poisson distribution

        skewness : float
            skewness parameter for the Poisson distribution

        Returns
        ----------
        - Adjusted score of the Poisson family
        """
        return (y-mean)/float(mean)

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS Poisson Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Poisson distribution

        scale : float
            scale parameter for the Poisson distribution

        shape : float
            tail thickness parameter for the Poisson distribution

        skewness : float
            skewness parameter for the Poisson distribution

        Returns
        ----------
        - Score of the Poisson family
        """
        return X*(y-mean)

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS Poisson Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Poisson model - gradient only
        """
        return gas_recursion_poisson_orderone

    @staticmethod
    def newton_recursion():
        """ GAS Poisson Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Poisson model - adjusted score
        """
        return gas_recursion_poisson_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX Poisson Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX Poisson model - gradient only
        """
        return gasx_recursion_poisson_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX Poisson Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX Poisson model - adjusted score
        """
        return gasx_recursion_poisson_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level Poisson Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level Poisson model - gradient only
        """
        return gas_llev_recursion_poisson_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level Poisson Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level Poisson model - adjusted score
        """
        return gas_llev_recursion_poisson_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend Poisson Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Poisson model - gradient only
        """
        return gas_llt_recursion_poisson_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend Poisson Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Poisson model - adjusted score
        """
        return gas_llt_recursion_poisson_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression Poisson Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Poisson model - gradient only
        """
        return gas_reg_recursion_poisson_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression Poisson Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Poisson model - adjusted score
        """
        return gas_reg_recursion_poisson_ordertwo


class GASt(GASDistribution):
    """ GAS t Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "t GAS"
        link = np.array
        scale = True
        shape = True
        skewness = False
        mean_transform = np.array
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['t Scale', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 0.01])
        lvs_to_build.append(['v', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 2.5])
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from t distribution

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return loc + scale*np.random.standard_t(shape,nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS t Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the t distribution

        scale : float
            scale parameter for the t distribution

        shape : float
            tail thickness parameter for the t distribution

        skewness : float
            skewness parameter for the t distribution

        Returns
        ----------
        - Score of the t family
        """
        return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(y-mean,2)/shape))

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the t distribution

        scale : float
            scale parameter for the t distribution

        shape : float
            tail thickness parameter for the t distribution

        skewness : float
            skewness parameter for the t distribution

        Returns
        ----------
        - Negative loglikelihood of the t family
        """
        return -np.sum(ss.t.logpdf(x=y, df=shape, loc=mean, scale=scale))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS t Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the t distribution

        scale : float
            scale parameter for the t distribution

        shape : float
            tail thickness parameter for the t distribution

        skewness : float
            skewness parameter for the t distribution

        Returns
        ----------
        - Adjusted score of the t family
        """
        return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(y-mean,2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(y-mean,2))/np.power((np.power(scale,2)*shape) + np.power(y-mean,2),2))

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS t Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the t distribution

        scale : float
            scale parameter for the t distribution

        shape : float
            tail thickness parameter for the t distribution

        skewness : float
            skewness parameter for the t distribution

        Returns
        ----------
        - Score of the t family
        """
        return ((shape+1)/shape)*((y-mean)*X)/(np.power(scale,2)+np.power((y-mean),2)/shape)

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS t model - gradient only
        """
        return gas_recursion_t_orderone

    @staticmethod
    def newton_recursion():
        """ GAS t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS t model - adjusted score
        """
        return gas_recursion_t_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX t model - gradient only
        """
        return gasx_recursion_t_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX t model - adjusted score
        """
        return gasx_recursion_t_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level t model - gradient only
        """
        return gas_llev_recursion_t_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level t model - adjusted score
        """
        return gas_llev_recursion_t_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend t model - gradient only
        """
        return gas_llt_recursion_t_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend t model - adjusted score
        """
        return gas_llt_recursion_t_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression t model - gradient only
        """
        return gas_reg_recursion_t_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression t model - adjusted score
        """
        return gas_reg_recursion_t_ordertwo


class GASSkewt(GASDistribution):
    """ GAS Skew-t Distribution

    Parameters
    ----------
    gradient_only : boolean
        Whether to use the gradient only as the update term, or use second-order information also
    """
    def __init__(self, gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score
        self.gradient_only = gradient_only

    @staticmethod
    def setup():
        """ Returns the attributes of this family

        Notes
        ----------
        - scale notes whether family has a variance parameter (sigma)
        - shape notes whether family has a tail thickness parameter (nu)
        - skewness notes whether family has a skewness parameter (gamma)
        - mean_transform is a function which transforms the location parameter
        - cythonized notes whether the family has cythonized routines
        
        Returns
        ----------
        - model name, link function, scale, shape, skewness, mean_transform, cythonized
        """
        name = "Skewt GAS"
        link = np.array
        scale = True
        shape = True
        skewness = True
        mean_transform = np.array
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Skewness', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 0.0])
        lvs_to_build.append(['Skewt Scale', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 0.01])
        lvs_to_build.append(['v', ifr.Uniform(transform='exp'), dst.q_Normal(0, 3), 2.5])
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from Skew t distribution

        Parameters
        ----------
        loc : float
            location parameter for the distribution

        scale : float
            scale parameter for the distribution

        shape : float
            tail thickness parameter for the distribution

        skewness : float
            skewness parameter for the distribution

        nsims : int or list
            number of draws to take from the distribution

        Returns
        ----------
        - Random draws from the distribution
        """
        return loc + scale*dst.skewt.rvs(shape, skewness, nsims)

    @staticmethod
    def first_order_score(y, mean, scale, shape, skewness):
        """ GAS Skew t Update term using gradient only - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Skew t distribution

        scale : float
            scale parameter for the Skew t distribution

        shape : float
            tail thickness parameter for the Skew t distribution

        skewness : float
            skewness parameter for the Skew t distribution

        Returns
        ----------
        - Score of the Skew t family
        """
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*(y-mean)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Skew t distribution

        scale : float
            scale parameter for the Skew t distribution

        shape : float
            tail thickness parameter for the Skew t distribution

        skewness : float
            skewness parameter for the Skew t distribution

        Returns
        ----------
        - Negative loglikelihood of the Skew t family
        """
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        return -np.sum(dst.skewt.logpdf(x=y,df=shape,loc=mean,gamma=skewness,scale=scale))

    @staticmethod
    def second_order_score(y, mean, scale, shape, skewness):
        """ GAS Skew t Update term potentially using second-order information - native Python function

        Parameters
        ----------
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Skew t distribution

        scale : float
            scale parameter for the Skew t distribution

        shape : float
            tail thickness parameter for the Skew t distribution

        skewness : float
            skewness parameter for the Skew t distribution

        Returns
        ----------
        - Adjusted score of the Skew t family
        """
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*(y-mean)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))

    @staticmethod
    def reg_score_function(X, y, mean, scale, shape, skewness):
        """ GAS Skew t Regression Update term using gradient only - native Python function

        Parameters
        ----------
        X : float
            datapoint for the right hand side variable
    
        y : float
            datapoint for the time series

        mean : float
            location parameter for the Skew t distribution

        scale : float
            scale parameter for the Skew t distribution

        shape : float
            tail thickness parameter for the Skew t distribution

        skewness : float
            skewness parameter for the Skew t distribution

        Returns
        ----------
        - Score of the Skew t family
        """
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*((y-mean)*X)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*((y-mean)*X)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))

    # Optional Cythonized recursions below

    @staticmethod
    def gradient_recursion():
        """ GAS Skew t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Skew t model - gradient only
        """
        return gas_recursion_skewt_orderone

    @staticmethod
    def newton_recursion():
        """ GAS Skew t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Skew t model - adjusted score
        """
        return gas_recursion_skewt_ordertwo

    @staticmethod
    def gradientx_recursion():
        """ GASX Skew t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GASX Skew t model - gradient only
        """
        return gasx_recursion_skewt_orderone

    @staticmethod
    def newtonx_recursion():
        """ GASX Skew t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GASX Skew t model - adjusted score
        """
        return gasx_recursion_skewt_ordertwo

    @staticmethod
    def gradientllev_recursion():
        """ GAS Local Level Skew t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Level Skew t model - gradient only
        """
        return gas_llev_recursion_skewt_orderone

    @staticmethod
    def newtonllev_recursion():
        """ GAS Local Level Skew t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Level Skew t model - adjusted score
        """
        return gas_llev_recursion_skewt_ordertwo

    @staticmethod
    def gradientllt_recursion():
        """ GAS Local Linear Trend Skew t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Skew t model - gradient only
        """
        return gas_llt_recursion_skewt_orderone

    @staticmethod
    def newtonllt_recursion():
        """ GAS Local Linear Trend Skew t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Local Linear Trend Skew t model - adjusted score
        """
        return gas_llt_recursion_skewt_ordertwo

    @staticmethod
    def gradientreg_recursion():
        """ GAS Dynamic Regression Skew t Model Recursion - gradient only

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Skew t model - gradient only
        """
        return gas_reg_recursion_skewt_orderone

    @staticmethod
    def newtonreg_recursion():
        """ GAS Dynamic Regression Skew t Model Recursion - adjusted score

        Returns
        ----------
        - Recursion function for GAS Dynamic Regression Skew t model - adjusted score
        """
        return gas_reg_recursion_skewt_ordertwo