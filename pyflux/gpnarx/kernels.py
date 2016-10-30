import sys
import numpy as np

from .. import families as fam

from .kernel_routines import SE_K_matrix, SE_Kstar_matrix, SE_Kstarstar_matrix, SE_K_arbitrary_X_matrix
from .kernel_routines import OU_K_matrix, OU_Kstar_matrix, OU_Kstarstar_matrix, OU_K_arbitrary_X_matrix
from .kernel_routines import RQ_K_matrix, RQ_Kstar_matrix, RQ_Kstarstar_matrix, RQ_K_arbitrary_X_matrix
from .kernel_routines import ARD_K_matrix, ARD_Kstar_matrix, ARD_Kstarstar_matrix, ARD_K_arbitrary_X_matrix
from .kernel_routines import Periodic_K_matrix, Periodic_Kstar_matrix, Periodic_Kstarstar_matrix, Periodic_K_arbitrary_X_matrix


class SquaredExponential(object):
    """ Squared Exponential Kernel

    Parameters
    ----------
    X : np.ndarray
        The RHS factors for the GP regression
    """

    def __init__(self, X=np.array([1])):
        self.X = X.transpose()

    @staticmethod
    def build_latent_variables():
        """ Builds latent variables for this kernel

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['l', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['tau', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        return SE_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        return SE_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        return SE_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        return SE_Kstarstar_matrix(Xstar, parm)


class OrnsteinUhlenbeck(object):
    """ Ornstein Uhlenbeck Kernel

    Parameters
    ----------
    X : np.ndarray
        The RHS factors for the GP regression
    """

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    @staticmethod
    def build_latent_variables():
        """ Builds latent variables for this kernel

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['l', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['tau', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        """ Returns the Gram Matrix

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the Gram Matrix

        Returns
        ----------
        - Gram Matrix (np.ndarray)
        """
        return OU_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        """ Returns K(x1,x2)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x1, x2)

        Xstar1 : np.ndarray
            First data subset

        Xstar2 : np.ndarray
            Second data subset

        Returns
        ----------
        - K(x1, x2)
        """
        return OU_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        """ Returns K(x, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x, x*)
        """
        return OU_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        """ Returns K(x*, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x*, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x*, x*)
        """
        return OU_Kstarstar_matrix(Xstar, parm)


class ARD(object):
    """ ARD Kernel

    Parameters
    ----------
    X : np.ndarray
        The RHS factors for the GP regression
    """

    def __init__(self, X=np.array([1])):
        self.X = X.transpose()

    def build_latent_variables(self):
        """ Builds latent variables for this kernel

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        for lag in range(self.X.shape[1]):
            lvs_to_build.append(['l lag' + str(lag+1), fam.FLat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['tau', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        """ Returns the Gram Matrix

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the Gram Matrix

        Returns
        ----------
        - Gram Matrix (np.ndarray)
        """
        return ARD_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        """ Returns K(x1,x2)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x1, x2)

        Xstar1 : np.ndarray
            First data subset

        Xstar2 : np.ndarray
            Second data subset

        Returns
        ----------
        - K(x1, x2)
        """
        return ARD_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        """ Returns K(x, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x, x*)
        """
        return ARD_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        """ Returns K(x*, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x*, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x*, x*)
        """
        return ARD_Kstarstar_matrix(Xstar, parm)


class RationalQuadratic(object):
    """ Rational Quadratic Kernel

    Parameters
    ----------
    X : np.ndarray
        The RHS factors for the GP regression
    """

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    def build_latent_variables(self):
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['a', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['l', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['tau', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        """ Returns the Gram Matrix

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the Gram Matrix

        Returns
        ----------
        - Gram Matrix (np.ndarray)
        """
        return RQ_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        """ Returns K(x1,x2)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x1, x2)

        Xstar1 : np.ndarray
            First data subset

        Xstar2 : np.ndarray
            Second data subset

        Returns
        ----------
        - K(x1, x2)
        """
        return RQ_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        """ Returns K(x, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x, x*)
        """
        return RQ_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        """ Returns K(x*, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x*, x*)

        Xstar : np.ndarray
            Data for prediction
        
        Returns
        ----------
        - K(x*, x*)
        """
        return RQ_Kstarstar_matrix(Xstar, parm)


class Periodic(object):
    """ Periodic Kernel

    Parameters
    ----------
    X : np.ndarray
        The RHS factors for the GP regression
    """

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    def build_latent_variables(self):
        """ Builds latent variables for this kernel

        Returns
        ----------
        - A list of lists (each sub-list contains latent variable information)
        """
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['l', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        lvs_to_build.append(['tau', fam.Flat(transform='exp'), fam.Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        """ Returns the Gram Matrix

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the Gram Matrix

        Returns
        ----------
        - Gram Matrix (np.ndarray)
        """
        return Periodic_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        """ Returns K(x1,x2)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x1, x2)

        Xstar1 : np.ndarray
            First data subset

        Xstar2 : np.ndarray
            Second data subset

        Returns
        ----------
        - K(x1, x2)
        """
        return Periodic_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        """ Returns K(x, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x, x*)
        """
        return Periodic_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        """ Returns K(x*, x*)

        Parameters
        ----------
        parm : np.ndarray
            Parameters for the K(x*, x*)

        Xstar : np.ndarray
            Data for prediction

        Returns
        ----------
        - K(x*, x*)
        """
        return Periodic_Kstarstar_matrix(Xstar, parm)