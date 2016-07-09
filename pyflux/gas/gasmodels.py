import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as sp

from .. import inference as ifr
from .. import tsm as tsm
from .. import distributions as dst
from .. import data_check as dc

def exponential_link(x):
    return 1.0/np.exp(x)

class GASDistribution(object):

    def __init__(self,gradient_only=False):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score


class GASExponential(GASDistribution):

    def __init__(self,gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    @staticmethod
    def setup():
        name = "Exponential GAS"
        link = exponential_link
        scale = False
        shape = False
        skewness = False
        mean_transform = np.log
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return np.random.exponential(1.0/loc, nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        return 1 - (mean*y)

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        return -np.sum(ss.expon.logpdf(x=y,scale=1/mean))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        return 1 - (mean*y)

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        return X*(1.0 - mean*y)


class GASLaplace(GASDistribution):

    def __init__(self,gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    @staticmethod
    def setup():
        name = "Laplace GAS"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        parameters_to_build.append(['Laplace Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),2.0])
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return np.random.laplace(loc, scale, nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        return (y-mean)/float(scale*np.abs(y-mean))

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        return -np.sum(ss.laplace.logpdf(y,loc=mean,scale=scale))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        return ((y-mean)/float(scale*np.abs(y-mean))) / (-(np.power(y-mean,2) - np.power(np.abs(mean-y),2))/(scale*np.power(np.abs(mean-y),3)))

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        return X*(y-mean)/(scale*np.abs(y-mean))


class GASNormal(GASDistribution):

    @staticmethod
    def setup():
        name = "Normal GAS"
        link = np.array
        scale = True
        shape = False
        skewness = False
        mean_transform = np.array
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        parameters_to_build.append(['Normal Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),0.0])
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return np.random.normal(loc, scale, nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        return (y-mean)/np.power(scale,2)

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        return -np.sum(ss.norm.logpdf(y,loc=mean,scale=scale))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        return y-mean

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        return X*(y-mean)


class GASPoisson(GASDistribution):

    @staticmethod
    def setup():
        name = "Poisson GAS"
        link = np.exp
        scale = False
        shape = False
        skewness = False
        mean_transform = np.log
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return np.random.poisson(loc, nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        return y-mean

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        return -np.sum(ss.poisson.logpmf(y,mean))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        return (y-mean)/float(mean)

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        return X*(y-mean)


class GASt(GASDistribution):

    def __init__(self,gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    @staticmethod
    def setup():
        name = "t GAS"
        link = np.array
        scale = True
        shape = True
        skewness = False
        mean_transform = np.array
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        parameters_to_build.append(['t Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),0.0])
        parameters_to_build.append(['v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),2.0])
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return loc + scale*np.random.standard_t(shape,nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(y-mean,2)/shape))

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        return -np.sum(ss.t.logpdf(x=y,df=shape,loc=mean,scale=scale))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(y-mean,2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(y-mean,2))/np.power((np.power(scale,2)*shape) + np.power(y-mean,2),2))

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        return ((shape+1)/shape)*((y-mean)*X)/(np.power(scale,2)+np.power((y-mean),2)/shape)


class GASSkewt(GASDistribution):

    def __init__(self,gradient_only=True):
        if gradient_only is True:
            self.score_function = self.first_order_score
        else:
            self.score_function = self.second_order_score

    @staticmethod
    def setup():
        name = "Skewt GAS"
        link = np.array
        scale = True
        shape = True
        skewness = True
        mean_transform = np.array
        return name, link, scale, shape, skewness, mean_transform

    @staticmethod
    def build_parameters():
        parameters_to_build = []
        parameters_to_build.append(['Skewness',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),0.0])
        parameters_to_build.append(['Skewt Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),0.0])
        parameters_to_build.append(['v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3),2.0])
        return parameters_to_build

    @staticmethod
    def draw_variable(loc,scale,shape,skewness,nsims):
        return loc + scale*dst.skewt.rvs(shape,skewness,nsims)

    @staticmethod
    def first_order_score(y,mean,scale,shape,skewness):
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*(y-mean)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))

    @staticmethod
    def neg_loglikelihood(y,mean,scale,shape,skewness):
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        return -np.sum(dst.skewt.logpdf(x=y,df=shape,loc=mean,gamma=skewness,scale=scale))

    @staticmethod
    def second_order_score(y,mean,scale,shape,skewness):
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*(y-mean)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*(y-mean)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))

    @staticmethod
    def reg_score_function(X,y,mean,scale,shape,skewness):
        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(shape/2.0))
        mean = mean + (skewness - (1.0/skewness))*scale*m1
        if (y-mean)>=0:
            return ((shape+1)/shape)*((y-mean)*X)/(np.power(skewness*scale,2) + (np.power(y-mean,2)/shape))
        else:
            return ((shape+1)/shape)*((y-mean)*X)/(np.power(scale,2) + (np.power(skewness*(y-mean),2)/shape))


