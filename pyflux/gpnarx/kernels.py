import sys
import numpy as np

from .. import distributions as dst
from .. import inference as ifr

from .kernel_routines import SE_K_matrix, SE_Kstar_matrix, SE_Kstarstar_matrix, SE_K_arbitrary_X_matrix
from .kernel_routines import OU_K_matrix, OU_Kstar_matrix, OU_Kstarstar_matrix, OU_K_arbitrary_X_matrix
from .kernel_routines import RQ_K_matrix, RQ_Kstar_matrix, RQ_Kstarstar_matrix, RQ_K_arbitrary_X_matrix
from .kernel_routines import ARD_K_matrix, ARD_Kstar_matrix, ARD_Kstarstar_matrix, ARD_K_arbitrary_X_matrix
from .kernel_routines import Periodic_K_matrix, Periodic_Kstar_matrix, Periodic_Kstarstar_matrix, Periodic_K_arbitrary_X_matrix


class SquaredExponential(object):

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    @staticmethod
    def build_latent_variables():
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['l', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['tau', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
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

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    @staticmethod
    def build_latent_variables():
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['l', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['tau', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        return OU_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        return OU_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        return OU_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        return OU_Kstarstar_matrix(Xstar, parm)


class ARD(object):

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    def build_latent_variables(self):
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        for lag in range(self.X.shape[1]):
            lvs_to_build.append(['l lag' + str(lag+1), ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['tau', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        return ARD_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        return ARD_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        return ARD_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        return ARD_Kstarstar_matrix(Xstar, parm)


class RationalQuadratic(object):

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    def build_latent_variables(self):
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['a', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['l', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['tau', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        return RQ_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        return RQ_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        return RQ_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        return RQ_Kstarstar_matrix(Xstar, parm)


class Periodic(object):

    def __init__(self, X=np.array([1])):

        self.X = X.transpose()

    def build_latent_variables(self):
        lvs_to_build = []
        lvs_to_build.append(['Noise Sigma^2', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['l', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        lvs_to_build.append(['tau', ifr.Uniform(transform='exp'), dst.q_Normal(0,3), -1.0])
        return lvs_to_build

    def K(self, parm):
        return Periodic_K_matrix(self.X, parm) + np.identity(self.X.shape[0])*(10**-10)

    def K_arbitrary_X(self, parm, Xstar1, Xstar2):
        return Periodic_K_arbitrary_X_matrix(Xstar1, Xstar2, parm)

    def Kstar(self, parm, Xstar):
        return Periodic_Kstar_matrix(self.X, Xstar, parm)

    def Kstarstar(self, parm, Xstar):
        return Periodic_Kstarstar_matrix(Xstar, parm)