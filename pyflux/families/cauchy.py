import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .. import families as fam

from .family import Family


class Cauchy(Family):
    """ 
    Cauchy Distribution
    ----
    This class contains methods relating to the Cauchy distribution for time series.
    """

    def __init__(self, loc=0.0, scale=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        loc : float
            Location parameter for the Cauchy distribution

        sigma : float
            Dispersion parameter for the Cauchy distribution

        transform : str
            Whether to apply a transformation - e.g. 'exp' or 'logit'
        """
        super(Cauchy, self).__init__(transform)
        self.loc0 = loc
        self.scale0 = scale
        self.covariance_prior = False

    def approximating_model(self, beta, T, Z, R, Q, h_approx, data):
        """ Creates approximating Gaussian state space model for the Cauchy measurement density
        
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
        """ Creates approximating Gaussian state space model for the Cauchy measurement density
        
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
        lvs_to_build.append(['Cauchy Scale', fam.Flat(transform='exp'), fam.Normal(0, 3), 0.0])
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
        return ss.cauchy.rvs(loc, scale, nsims)

    def logpdf(self, mu):
        """
        Log PDF for Cauchy prior

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
        return ss.cauchy.logpdf(mu, self.loc0, self.scale0)

    def pdf(self, mu):
        """
        PDF for Cauchy prior

        Parameters
        ----------
        mu : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - p(mu)
        """
        return ss.cauchy.pdf(mu, self.loc0, self.scale0)

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for each likelihood term - used for state space models

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Cauchy distribution

        scale : float
            scale parameter for the Cauchy distribution

        shape : float
            tail thickness parameter for the Cauchy distribution

        skewness : float
            skewness parameter for the Cauchy distribution

        Returns
        ----------
        - Markov blanket of the Cauchy family
        """
        return ss.cauchy.logpdf(y, loc=mean, scale=scale)

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
        name = "Cauchy"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        cythonized = False # used for GAS models
        return name, link, scale, shape, skewness, mean_transform, cythonized

    @staticmethod
    def neg_loglikelihood(y, mean, scale, shape, skewness):
        """ Negative loglikelihood function for this distribution

        Parameters
        ----------
        y : np.ndarray
            univariate time series

        mean : np.ndarray
            array of location parameters for the Cauchy distribution

        scale : float
            scale parameter for the Cauchy distribution

        shape : float
            tail thickness parameter for the Cauchy distribution

        skewness : float
            skewness parameter for the Cauchy distribution

        Returns
        ----------
        - Negative loglikelihood of the Cauchy family
        """
        return -np.sum(ss.cauchy.logpdf(y, loc=mean, scale=scale))


