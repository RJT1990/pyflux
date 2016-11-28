import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family
from .flat import Flat
from .normal import Normal

from .gas_recursions import gas_recursion_exponential_orderone, gas_recursion_exponential_ordertwo
from .gas_recursions import gasx_recursion_exponential_orderone, gasx_recursion_exponential_ordertwo
from .gas_recursions import gas_llev_recursion_exponential_orderone, gas_llev_recursion_exponential_ordertwo
from .gas_recursions import gas_llt_recursion_exponential_orderone, gas_llt_recursion_exponential_ordertwo
from .gas_recursions import gas_reg_recursion_exponential_orderone, gas_reg_recursion_exponential_ordertwo


class Exponential(Family):
    """ 
    Exponential Distribution
    ----
    This class contains methods relating to the Exponential distribution for time series.
    """

    def __init__(self, lmd=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        lambda : float
            Rate parameter for the Exponential distribution

        transform : str
            Whether to apply a transformation to the location variable - e.g. 'exp' or 'logit'
        """
        super(Exponential, self).__init__(transform)
        self.lmd0 = lmd
        self.covariance_prior = False

        self.gradient_only = kwargs.get('gradient_only', False) # used for GAS Exponential models
        if self.gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    def approximating_model(self, beta, T, Z, R, Q, h_approx, data):
        """ Creates approximating Gaussian state space model for Exponential measurement density
        
        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables
        
        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm
        
        h_approx : float
            The variance of the measurement density

        data: np.array
            The univariate time series data
        
        Returns
        ----------
        H : np.array
            Approximating measurement variance matrix
        
        mu : np.array
            Approximating measurement constants
        """     

        H = np.ones(data.shape[0])*h_approx
        mu = np.zeros(data.shape[0])

        return H, mu


    def approximating_model_reg(self, beta, T, Z, R, Q, h_approx, data, X, state_no):
        """ Creates approximating Gaussian state space model for Exponential measurement density
        
        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables
        
        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm
        
        h_approx : float
            The variance of the measurement density

        data: np.array
            The univariate time series data
        
        X: np.array
            The regressors

        state_no : int
            Number of states

        Returns
        ----------
        H : np.array
            Approximating measurement variance matrix
        
        mu : np.array
            Approximating measurement constants
        """     

        H = np.ones(data.shape[0])*h_approx
        mu = np.zeros(data.shape[0])

        return H, mu

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

    def logpdf(self, mu):
        """
        Log PDF for Exponential prior

        Parameters
        ----------
        mu : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - log(p(mu))
        """
        if self.transform is not None:
            mu = self.transform(mu)    
        return ss.expon.logpdf(mu, self.lmd0) 

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for the Exponential distribution

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
        - Markov blanket of the Exponential family
        """
        return ss.expon.logpdf(x=y, scale=1/mean)

    @staticmethod
    def exponential_link(x):
        return 1.0/np.exp(x)

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
        link = Exponential.exponential_link
        scale = False
        shape = False
        skewness = False
        mean_transform = np.log
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

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

    def pdf(self, mu):
        """
        PDF for Exponential prior

        Parameters
        ----------
        mu : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - p(mu)
        """
        if self.transform is not None:
            mu = self.transform(mu)                
        return ss.expon.pdf(mu, self.lmd0) 

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

    # Optional Cythonized recursions below for GAS Exponential models

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