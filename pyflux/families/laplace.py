import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family
from .flat import Flat
from .normal import Normal

from .gas_recursions import gas_recursion_laplace_orderone, gas_recursion_laplace_ordertwo
from .gas_recursions import gasx_recursion_laplace_orderone, gasx_recursion_laplace_ordertwo
from .gas_recursions import gas_llev_recursion_laplace_orderone, gas_llev_recursion_laplace_ordertwo
from .gas_recursions import gas_llt_recursion_laplace_orderone, gas_llt_recursion_laplace_ordertwo
from .gas_recursions import gas_reg_recursion_laplace_orderone, gas_reg_recursion_laplace_ordertwo


class Laplace(Family):
    """ 
    Laplace Distribution
    ----
    This class contains methods relating to the Laplace distribution for time series.
    """

    def __init__(self, loc=0.0, scale=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        loc : float
            Location parameter for the Laplace distribution

        scale : float
            Scale parameter for the Laplace distribution

        transform : str
            Whether to apply a transformation to the location variable - e.g. 'exp' or 'logit'
        """
        super(Laplace, self).__init__(transform)
        self.loc0 = loc
        self.scale0 = scale
        self.covariance_prior = False

        self.gradient_only = kwargs.get('gradient_only', False) # used for GAS Laplace models
        if self.gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    def approximating_model(self, beta, T, Z, R, Q, h_approx, data):
        """ Creates approximating Gaussian state space model for Poisson measurement density
        
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
        """ Creates approximating Gaussian state space model for Poisson measurement density
        
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
        lvs_to_build.append(['Laplace Scale', Flat(transform='exp'), Normal(0, 3), 2.0])
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

    def logpdf(self, mu):
        """
        Log PDF for Laplace prior

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
        return ss.laplace.logpdf(mu, loc=self.loc0, scale=self.scale0) 

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for each likelihood term

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
        - Markov Blanket of the Laplace family
        """
        return ss.laplace.logpdf(y, loc=mean, scale=scale)

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
        name = "Laplace"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
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

    def pdf(self, mu):
        """
        PDF for Laplace prior

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
        return ss.laplace.pdf(mu, self.loc0, self.scale0) 

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

    # Optional Cythonized recursions below for GAS Laplace models

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
