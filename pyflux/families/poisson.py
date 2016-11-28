import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .poisson_kalman_recursions import nl_univariate_KFS, nld_univariate_KFS
from .family import Family
from .flat import Flat
from .normal import Normal

from .gas_recursions import gas_recursion_poisson_orderone, gas_recursion_poisson_ordertwo
from .gas_recursions import gasx_recursion_poisson_orderone, gasx_recursion_poisson_ordertwo
from .gas_recursions import gas_llev_recursion_poisson_orderone, gas_llev_recursion_poisson_ordertwo
from .gas_recursions import gas_llt_recursion_poisson_orderone, gas_llt_recursion_poisson_ordertwo
from .gas_recursions import gas_reg_recursion_poisson_orderone, gas_reg_recursion_poisson_ordertwo


class Poisson(Family):
    """ 
    Poisson Distribution
    ----
    This class contains methods relating to the Poisson distribution for time series.
    """

    def __init__(self, lmd=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        lambda : float
            Rate parameter for the Poisson distribution

        transform : str
            Whether to apply a transformation to the Poisson latent variable - e.g. 'exp' or 'logit'
        """
        super(Poisson, self).__init__(transform)
        self.lmd0 = lmd
        self.covariance_prior = False

        self.gradient_only = kwargs.get('gradient_only', False) # used for GAS Poisson models
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

        H = np.ones(data.shape[0])
        mu = np.zeros(data.shape[0])

        alpha = np.array([np.zeros(data.shape[0])])
        tol = 100.0
        it = 0
        while tol > 10**-7 and it < 5:
            old_alpha = alpha[0]
            alpha, V = nl_univariate_KFS(data,Z,H,T,Q,R,mu)
            H = np.exp(-alpha[0])
            mu = data - alpha[0] - np.exp(-alpha[0])*(data - np.exp(alpha[0]))
            tol = np.mean(np.abs(alpha[0]-old_alpha))
            it += 1

        return H, mu

    def approximating_model_reg(self, beta, T, Z, R, Q, h_approx, data, X, state_no):
        """ Creates approximating Gaussian model for Poisson measurement density
        - dynamic regression model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm

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
        
        H = np.ones(data.shape[0])
        mu = np.zeros(data.shape[0])

        alpha = np.zeros([state_no, data.shape[0]])
        tol = 100.0
        it = 0
        while tol > 10**-7 and it < 5:
            old_alpha = np.sum(X*alpha.T,axis=1)
            alpha, V = nld_univariate_KFS(data,Z,H,T,Q,R,mu)
            H = np.exp(-np.sum(X*alpha.T,axis=1))
            mu = data - np.sum(X*alpha.T,axis=1) - np.exp(-np.sum(X*alpha.T,axis=1))*(data - np.exp(np.sum(X*alpha.T,axis=1)))
            tol = np.mean(np.abs(np.sum(X*alpha.T,axis=1)-old_alpha))
            it += 1

        return H, mu

    @staticmethod
    def build_latent_variables():
        """ Builds additional latent variables for this family in a probabilistic model

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

    def logpdf(self, mu):
        """
        Log PDF for Poisson prior

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
        return ss.poisson.logpmf(mu, self.lmd0) 

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for the Poisson distribution

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
        - Markov blanket of the Poisson family
        """
        return ss.poisson.logpmf(y, mean)

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
        name = "Poisson"
        link = np.exp
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
        return -np.sum(-mean + np.log(mean)*y - sp.gammaln(y + 1))

    def pdf(self, mu):
        """
        PDF for Poisson prior

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
        return ss.poisson.pmf(mu, self.lmd0) 

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

    # Optional Cythonized recursions below for GAS Poisson models

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