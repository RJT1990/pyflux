import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family


class InverseGamma(Family):
    """ 
    Inverse Gamma Distribution
    ----
    This class contains methods relating to the inverse gamma distribution for time series.
    """

    def __init__(self, alpha, beta, transform=None, **kwargs):
        """
        Parameters
        ----------
        alpha : float
            Alpha parameter for the Inverse Gamma distribution

        beta : float
            Beta parameter for the Inverse Gamma distribution

        transform : str
            Whether to apply a transformation - e.g. 'exp' or 'logit'
        """
        super(InverseGamma, self).__init__(transform)
        self.covariance_prior = False
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        """
        Log PDF for Inverse Gamma prior

        Parameters
        ----------
        x : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - log(p(x))
        """
        if self.transform is not None:
            x = self.transform(x)       
        return (-self.alpha-1)*np.log(x) - (self.beta/float(x))

    def pdf(self, x):
        """
        PDF for Inverse Gamma prior

        Parameters
        ----------
        x : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - p(x)
        """
        if self.transform is not None:
            x = self.transform(x)               
        return (x**(-self.alpha-1))*np.exp(-(self.beta/float(x)))