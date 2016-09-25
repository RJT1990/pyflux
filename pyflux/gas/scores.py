from numpy import abs, exp, power, array, sqrt, pi
from scipy.special import gamma

#TODO: This file should eventually be replaced, by moving the existing functions (used for GARCH based models)
# into the GARCH folder, with Cythonizations

class Score(object):

    @staticmethod
    def score(y,loc,scale,shape):
        pass

    @staticmethod
    def adj_score(y,loc,scale,shape):
        pass

class BetatScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def mu_score(y,loc,scale,shape):
        try:
            return (((shape+1.0)*power(y-loc,2))/float(shape*exp(scale) + power(y-loc,2))) - 1.0
        except:
            return -1.0

    @staticmethod
    def mu_adj_score(y,loc,scale,shape):
        try:
            return (((shape+1.0)*power(y-loc,2))/float(shape*exp(scale) + power(y-loc,2))) - 1.0
        except:
            return -1.0

    def score(self,y,loc,scale,shape):
        return array([self.mu_score(y,loc,scale,shape)])

    def adj_score(self,y,loc,scale,shape):
        return array([self.mu_adj_score(y,loc,scale,shape)])

class SkewBetatScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def tv_variate_exp(df):
        return (sqrt(df)*gamma((df-1.0)/2.0))/(sqrt(pi)*gamma(df/2.0))

    @staticmethod
    def mu_score(y,loc,scale,shape,skewness):
        try:
            if (y-loc)>=0:
                return (((shape+1.0)*power(y-loc,2))/float(power(skewness,2)*shape*exp(scale) + power(y-loc,2))) - 1.0
            else:
                return (((shape+1.0)*power(y-loc,2))/float(power(skewness,-2)*shape*exp(scale) + power(y-loc,2))) - 1.0    
        except:
            return -1.0

    @staticmethod
    def mu_adj_score(y,loc,scale,shape,skewness):
        try:
            if (y-loc)>=0:
                return (((shape+1.0)*power(y-loc,2))/float(power(skewness,2)*shape*exp(scale) + power(y-loc,2))) - 1.0
            else:
                return (((shape+1.0)*power(y-loc,2))/float(power(skewness,-2)*shape*exp(scale) + power(y-loc,2))) - 1.0    
        except:
            return -1.0

    def score(self,y,loc,scale,shape,skewness):
        return array([self.mu_score(y,loc,scale,shape,skewness)])

    def adj_score(self,y,loc,scale,shape,skewness):
        return array([self.mu_adj_score(y,loc,scale,shape,skewness)])