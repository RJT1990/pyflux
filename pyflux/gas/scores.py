from numpy import abs, exp, power, array, sqrt, pi
from scipy.special import gamma

class Score(object):

    @staticmethod
    def score(y,loc,scale,shape):
        pass

    @staticmethod
    def adj_score(y,loc,scale,shape):
        pass


class ExponentialScore(Score):
    """ Exponential Score

    For dynamic parameter \log{\lambda_{t}}
    """

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def log_lam_score(y,lam):
        return 1 - (lam*y)

    @staticmethod
    def log_lam_adj_score(y,lam):
        return 1 - (lam*y)

    def score(self,y,lam):
        return array([self.log_lam_score(y,lam)])

    def adj_score(self,y,lam):
        return array([self.log_lam_adj_score(y,lam)])


class LaplaceScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def mu_score(y,loc,scale):
        return (y-loc)/float(scale*abs(y-loc))

    @staticmethod
    def mu_information_term(y,loc,scale):
        return -(power(y-loc,2) - power(np.abs(loc-y),2))/(scale*power(np.abs(loc-y),3))

    @staticmethod
    def mu_adj_score(y,loc,scale):
        return LaplaceScore.mu_score(y,loc,scale)/LaplaceScore.mu_information_term(y,loc,scale)

    def score(self,y,loc,scale):
        return array([self.mu_score(y,loc,scale)])

    def adj_score(self,y,loc,scale):
        return array([self.mu_adj_score(y,loc,scale)])


class Normal1Score(Score):
    """ Normal Score (1)

    For dynamic parameters \mu_{t} and \sigma^{2}_{t}
    """

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def mu_score(y,loc,var):
        return 0.5*(y-loc)/var

    @staticmethod
    def mu_adj_score(y,loc,var):
        return (y-loc)

    @staticmethod
    def var_score(y,loc,var):
        return -0.5/var + (0.5*power(y-loc,2))/power(var,2)

    @staticmethod
    def var_adj_score(y,loc,var):
        return power(y-loc,2) - var

    def score(self,y,loc,var):
        return array([self.mu_score(y,loc,var), self.var_score(y,loc,var)])

    def adj_score(self,y,loc,var):
        return array([self.mu_adj_score(y,loc,var), self.var_adj_score(y,loc,var)])


class Normal2Score(Score):
    """ Normal Score (2)

    For dynamic parameters \mu_{t} and \log{\sigma^{2}_{t}}
    """

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def mu_score(y,loc,var):
        return 0.5*(y-loc)/var

    @staticmethod
    def mu_adj_score(y,loc,var):
        return (y-loc)

    @staticmethod
    def log_var_score(y,loc,var):
        return -0.5 + (0.5*power(y-loc,2))/var

    @staticmethod
    def log_var_adj_score(y,loc,var):
        return (power(y-loc,2)/var) - 1

    def score(self,y,loc,var):
        return array([self.mu_score(y,loc,var), self.log_var_score(y,loc,var)])

    def adj_score(self,y,loc,var):
        return array([self.mu_adj_score(y,loc,var), self.log_var_adj_score(y,loc,var)])


class PoissonScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def log_lambda_score(y,lam):
        return (y-lam)

    @staticmethod
    def log_lambda_adj_score(y,lam):
        return (y-lam)/float(lam)

    def score(self,y,lam):
        return array([self.log_lam_score(y,lam)])

    def adj_score(self,y,lam):
        return array([self.log_lam_adj_score(y,lam)])


class SkewtScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def tv_variate_exp(df):
        return (sqrt(df)*gamma((df-1.0)/2.0))/(sqrt(pi)*gamma(df/2.0))

    @staticmethod
    def mu_score(y,loc,scale,shape,skewness):
        if (y-loc)>=0:
            return ((shape+1)/shape)*(y-loc)/(power(skewness*scale,2) + (power(y-loc,2)/shape))
        else:
            return ((shape+1)/shape)*(y-loc)/(power(scale,2) + (power(skewness*(y-loc),2)/shape))

    @staticmethod
    def mu_information_term(y,loc,scale,shape,skewness):
        #return (shape+1)*((power(scale,2)*shape) - power(y-loc,2))/power((power(scale,2)*shape) + power(y-loc,2),2)
        return 1.0

    @staticmethod
    def mu_adj_score(y,loc,scale,shape,skewness):
        return tScore.mu_score(y,loc,scale,shape,skewness) / tScore.mu_information_term(y,loc,scale,shape,skewness)

    def score(self,y,loc,scale,shape,skewness):
        return array([self.mu_score(y,loc,scale,shape,skewness)])

    def adj_score(self,y,loc,scale,shape,skewness):
        return array([self.mu_adj_score(y,loc,scale,shape,skewness)])


class tScore(Score):

    def __init__(self):

        super(Score,self).__init__()

    @staticmethod
    def mu_score(y,loc,scale,shape):
        return ((shape+1)/shape)*(y-loc)/(power(scale,2) + (power(y-loc,2)/shape))

    @staticmethod
    def mu_information_term(y,loc,scale,shape):
        return (shape+1)*((power(scale,2)*shape) - power(y-loc,2))/power((power(scale,2)*shape) + power(y-loc,2),2)

    @staticmethod
    def mu_adj_score(y,loc,scale,shape):
        return tScore.mu_score(y,loc,scale,shape) / tScore.mu_information_term(y,loc,scale,shape)

    def score(self,y,loc,scale,shape):
        return array([self.mu_score(y,loc,scale,shape)])

    def adj_score(self,y,loc,scale,shape):
        return array([self.mu_adj_score(y,loc,scale,shape)])

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