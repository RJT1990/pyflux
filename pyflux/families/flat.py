import numpy as np
import scipy.stats as ss
import scipy.special as sp

from .family import Family


class Flat(Family):
    """ 
    Flat Distribution
    ----
    This class contains methods relating to the flat prior distribution for time series.
    """

    def __init__(self, transform=None, **kwargs):
        """
        Parameters
        ----------
        transform : str
            Whether to apply a transformation - e.g. 'exp' or 'logit'
        """
        super(Flat, self).__init__(transform)
        self.covariance_prior = False

    def logpdf(self, mu):
        """
        Log PDF for Flat prior

        Parameters
        ----------
        mu : float
            Latent variable for which the prior is being formed over

        Returns
        ----------
        - log(p(mu))
        """
        return 0.0
