import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family
from .flat import Flat

from .gas_recursions import gas_recursion_normal_orderone, gas_recursion_normal_ordertwo
from .gas_recursions import gasx_recursion_normal_orderone, gasx_recursion_normal_ordertwo
from .gas_recursions import gas_llev_recursion_normal_orderone, gas_llev_recursion_normal_ordertwo
from .gas_recursions import gas_llt_recursion_normal_orderone, gas_llt_recursion_normal_ordertwo
from .gas_recursions import gas_reg_recursion_normal_orderone, gas_reg_recursion_normal_ordertwo


class Normal(Family):
    """ 
    Normal Distribution
    ----
    This class contains methods relating to the normal distribution for time series.
    """

    def __init__(self, mu=0.0, sigma=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        mu : float
            Mean parameter for the Normal distribution

        sigma : float
            Standard deviation for the Normal distribution

        transform : str
            Whether to apply a transformation for the location latent variable - e.g. 'exp' or 'logit'
        """
        super(Normal, self).__init__(transform)
        self.mu0 = mu
        self.sigma0 = sigma
        self.param_no = 2
        self.covariance_prior = False

        self.gradient_only = kwargs.get('gradient_only', False) # used for GAS Normal models
        if self.gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    def approximating_model(self, beta, T, Z, R, Q, h_approx, data):
        """ Creates approximating Gaussian state space model for the Normal measurement density
        
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
        """ Creates approximating Gaussian state space model for the Normal measurement density
        
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
        """ Builds additional latent variables for this family in a probabilistic model

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Normal Scale', Flat(transform='exp'), Normal(0, 3), 0.0])
        return lvs_to_build

    @staticmethod
    def draw_variable(loc, scale, shape, skewness, nsims):
        """ Draws random variables from this distribution with new latent variables

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

    def draw_variable_local(self, size): 
        """ Simulate from the Normal distribution using instance values

        Parameters
        ----------
        size : int
            How many simulations to perform

        Returns
        ----------
        np.ndarray of Normal random variable
        """
        return ss.norm.rvs(loc=self.mu0, scale=self.sigma0, size=size)

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

    def logpdf(self, mu):
        """
        Log PDF for Normal prior

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
        return -np.log(float(self.sigma0)) - (0.5*(mu-self.mu0)**2)/float(self.sigma0**2)

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for each likelihood term - used for state space models

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
        - Markov blanket of the Normal family
        """
        return ss.norm.logpdf(y, loc=mean, scale=scale)

    @staticmethod
    def setup():
        """ Returns the attributes of this family if using in a probabilistic model

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
        name = "Normal"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        cythonized = True
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function for this distribution

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

    def pdf(self, mu):
        """
        PDF for Normal prior

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
        return (1.0/float(self.sigma0))*np.exp(-(0.5*(mu-self.mu0)**2)/float(self.sigma0**2))

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

    def vi_change_param(self, index, value):
        """ Wrapper function for changing latent variables - variational inference

        Parameters
        ----------
        index : int
            0 or 1 depending on which latent variable

        value : float
            What to change the latent variable to
        """
        if index == 0:
            self.mu0 = value
        elif index == 1:
            self.sigma0 = np.exp(value)

    def vi_return_param(self, index):
        """ Wrapper function for selecting appropriate latent variable for variational inference

        Parameters
        ----------
        index : int
            0 or 1 depending on which latent variable

        Returns
        ----------
        The appropriate indexed parameter
        """
        if index == 0:
            return self.mu0
        elif index == 1:
            return np.log(self.sigma0)

    def vi_loc_score(self,x):
        """ The gradient of the location latent variable mu - used for variational inference

        Parameters
        ----------
        x : float
            A random variable

        Returns
        ----------
        The gradient of the location latent variable mu at x
        """
        return (x-self.mu0)/(self.sigma0**2)

    def vi_scale_score(self,x):
        """ The score of the scale, where scale = exp(x) - used for variational inference

        Parameters
        ----------
        x : float
            A random variable

        Returns
        ----------
        The gradient of the scale latent variable at x
        """
        return np.exp(-2.0*np.log(self.sigma0))*(x-self.mu0)**2 - 1.0

    def vi_score(self, x, index):
        """ Wrapper function for selecting appropriate score

        Parameters
        ----------
        x : float
            A random variable

        index : int
            0 or 1 depending on which latent variable

        Returns
        ----------
        The gradient of the scale latent variable at x
        """
        if index == 0:
            return self.vi_loc_score(x)
        elif index == 1:
            return self.vi_scale_score(x)


    # Optional Cythonized recursions below for GAS Normal models

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