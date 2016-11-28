import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family
from .flat import Flat
from .normal import Normal

from .gas_recursions import gas_recursion_skewt_orderone, gas_recursion_skewt_ordertwo
from .gas_recursions import gasx_recursion_skewt_orderone, gasx_recursion_skewt_ordertwo
from .gas_recursions import gas_llev_recursion_skewt_orderone, gas_llev_recursion_skewt_ordertwo
from .gas_recursions import gas_llt_recursion_skewt_orderone, gas_llt_recursion_skewt_ordertwo
from .gas_recursions import gas_reg_recursion_skewt_orderone, gas_reg_recursion_skewt_ordertwo


class Skewt(Family):
    """ 
    Student Skew t Distribution
    ----
    This class contains methods relating to the Student Skew t distribution for time series.
    """

    def __init__(self, loc=0.0, scale=1.0, df=8.0, gamma=1.0, transform=None, **kwargs):
        """
        Parameters
        ----------
        loc : float
            Location parameter for the Skew t distribution

        scale : float
            Scale parameter for the Skew t distribution

        df : float
            Degrees of freedom parameter for the Skew t distribution

        gamma : float
            Skewness parameter (1.0 is skewed; under 1.0, -ve skewed; over 1.0, +ve skewed)

        transform : str
            Whether to apply a transformation to the location variable - e.g. 'exp' or 'logit'
        """
        super(Skewt, self).__init__(transform)
        self.loc0 = loc
        self.scale0 = scale
        self.df0 = df
        self.gamma0 = gamma
        self.covariance_prior = False

        self.gradient_only = kwargs.get('gradient_only', False) # used for GAS t models
        if self.gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    def approximating_model(self, beta, T, Z, R, Q, h_approx, data):
        """ Creates approximating Gaussian state space model for Skewt measurement density
        
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
        """ Creates approximating Gaussian state space model for Skewt measurement density
        
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
        lvs_to_build.append(['Skewness', Flat(transform='exp'), Normal(0, 3), 0.0])
        lvs_to_build.append(['Skewt Scale', Flat(transform='exp'), Normal(0, 3), 0.01])
        lvs_to_build.append(['v', Flat(transform='exp'), Normal(0, 3), 2.5])
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
        return loc + scale*Skewt.rvs(shape, skewness, nsims)

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
    def rvs(df, gamma, n):
        """ Generates random variables from a Skew t distribution

        Parameters
        ----------  
        df : float
            degrees of freedom parameter

        gamma : float
            skewness parameter

        n : int or list
            Number of simulations to perform; if list input, produces array

        """

        if type(n) == list:
            u = np.random.uniform(size=n[0]*n[1])
            result = Skewt.ppf(q=u, df=df, gamma=gamma)
            result = np.split(result,n[0])
            return np.array(result)
        else:
            u = np.random.uniform(size=n)
            if isinstance(df, np.ndarray) or isinstance(gamma, np.ndarray):
                return np.array([Skewt.ppf(q=np.array([u[i]]), df=df[i], gamma=gamma[i])[0] for i in range(n)])
            else:
                return Skewt.ppf(q=u, df=df, gamma=gamma)

    @staticmethod
    def logpdf_internal(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        result = np.zeros(x.shape[0])
        result[x-loc<0] = np.log(2.0) - np.log(gamma + 1.0/gamma) + ss.t.logpdf(x=gamma*x[(x-loc) < 0], loc=loc[(x-loc) < 0]*gamma,df=df, scale=scale)
        result[x-loc>=0] = np.log(2.0) - np.log(gamma + 1.0/gamma) + ss.t.logpdf(x=x[(x-loc) >= 0]/gamma, loc=loc[(x-loc) >= 0]/gamma,df=df, scale=scale)
        return result

    @staticmethod
    def logpdf_internal_prior(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        if x-loc < 0.0:
            return np.log(2.0) - np.log(gamma + 1.0/gamma) + ss.t.logpdf(x=gamma*x, loc=loc*gamma,df=df, scale=scale)
        else:
            return np.log(2.0) - np.log(gamma + 1.0/gamma) + ss.t.logpdf(x=x/gamma, loc=loc/gamma,df=df, scale=scale)

    def logpdf(self, mu):
        """
        Log PDF for Skew t prior

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
        return self.logpdf_internal_prior(mu, df=self.df0, loc=self.loc0, scale=self.scale0, gamma=self.gamma0) 

    @staticmethod
    def markov_blanket(y, mean, scale, shape, skewness):
        """ Markov blanket for each likelihood term

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
        - Markov blanket of the Skew t family
        """
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        return Skewt.logpdf_internal(x=y, df=shape, loc=mean, gamma=skewness, scale=scale)

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
        name = "Skewt"
        link = np.array
        scale = True
        shape = True
        skewness = True
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
        return -np.sum(Skewt.logpdf_internal(x=y, df=shape, loc=mean, gamma=skewness, scale=scale))

    @staticmethod
    def pdf_internal(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        """
        Raw PDF function for the Skew t distribution
        """
        result = np.zeros(x.shape[0])
        result[x<0] = 2.0/(gamma + 1.0/gamma)*stats.t.pdf(x=gamma*x[(x-loc) < 0], loc=loc[(x-loc) < 0]*gamma,df=df, scale=scale)
        result[x>=0] = 2.0/(gamma + 1.0/gamma)*stats.t.pdf(x=x[(x-loc) >= 0]/gamma, loc=loc[(x-loc) >= 0]/gamma,df=df, scale=scale)
        return result

    def pdf(self, mu):
        """
        PDF for Skew t prior

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
        return self.pdf_internal(mu, df=self.df0, loc=self.loc0, scale=self.scale0, gamma=self.gamma0) 

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
    def cdf(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        """
        CDF function for Skew t distribution
        """
        result = np.zeros(x.shape[0])
        result[x<0] = 2.0/(np.power(gamma,2) + 1.0)*ss.t.cdf(gamma*(x[x-loc < 0]-loc[x-loc < 0])/scale, df=df)
        result[x>=0] = 1.0/(np.power(gamma,2) + 1.0) + 2.0/((1.0/np.power(gamma,2)) + 1.0)*(ss.t.cdf((x[x-loc >= 0]-loc[x-loc >= 0])/(gamma*scale), df=df)-0.5)
        return result

    @staticmethod
    def ppf(q, df, loc=0.0, scale=1.0, gamma = 1.0):
        """
        PPF function for Skew t distribution
        """
        result = np.zeros(q.shape[0])
        probzero = Skewt.cdf(x=np.zeros(1),loc=np.zeros(1),df=df,gamma=gamma)
        result[q<probzero] = 1.0/gamma*ss.t.ppf(((np.power(gamma,2) + 1.0) * q[q<probzero])/2.0,df)
        result[q>=probzero] = gamma*ss.t.ppf((1.0 + 1.0/np.power(gamma,2))/2.0*(q[q >= probzero] - probzero) + 0.5, df)
        return result

    # Optional Cythonized recursions below for GAS Skew t models

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
