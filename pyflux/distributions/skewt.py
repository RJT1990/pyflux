import numpy as np
from scipy import stats

class skewt(object):

    @staticmethod
    def logpdf(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        result = np.zeros(x.shape[0])
        result[x-loc<0] = np.log(2.0) - np.log(gamma + 1.0/gamma) + stats.t.logpdf(x=gamma*x[(x-loc) < 0], loc=loc[(x-loc) < 0]*gamma,df=df, scale=scale)
        result[x-loc>=0] = np.log(2.0) - np.log(gamma + 1.0/gamma) + stats.t.logpdf(x=x[(x-loc) >= 0]/gamma, loc=loc[(x-loc) >= 0]/gamma,df=df, scale=scale)
        return result

    @staticmethod
    def pdf(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        result = np.zeros(x.shape[0])
        result[x<0] = 2.0/(gamma + 1.0/gamma)*stats.t.pdf(x=gamma*x[(x-loc) < 0], loc=loc[(x-loc) < 0]*gamma,df=df, scale=scale)
        result[x>=0] = 2.0/(gamma + 1.0/gamma)*stats.t.pdf(x=x[(x-loc) >= 0]/gamma, loc=loc[(x-loc) >= 0]/gamma,df=df, scale=scale)
        return result

    @staticmethod
    def cdf(x, df, loc=0.0, scale=1.0, gamma = 1.0):
        result = np.zeros(x.shape[0])
        result[x<0] = 2.0/(np.power(gamma,2) + 1.0)*stats.t.cdf(gamma*(x[x-loc < 0]-loc[x-loc < 0])/scale, df=df)
        result[x>=0] = 1.0/(np.power(gamma,2) + 1.0) + 2.0/((1.0/np.power(gamma,2)) + 1.0)*(stats.t.cdf((x[x-loc >= 0]-loc[x-loc >= 0])/(gamma*scale), df=df)-0.5)
        return result

    @staticmethod
    def ppf(q, df, loc=0.0, scale=1.0, gamma = 1.0):
        result = np.zeros(q.shape[0])
        probzero = skewt.cdf(x=np.zeros(1),loc=np.zeros(1),df=df,gamma=gamma)
        result[q<probzero] = 1.0/gamma*stats.t.ppf(((np.power(gamma,2) + 1.0) * q[q<probzero])/2.0,df)
        result[q>=probzero] = gamma*stats.t.ppf((1.0 + 1.0/np.power(gamma,2))/2.0*(q[q >= probzero] - probzero) + 0.5, df)
        return result

    @staticmethod
    def rvs(df, gamma, n):
        u = np.random.uniform(size=n)
        if type(n) == list:
            result = []
            for i in range(n[0]):
                result.append(skewt.ppf(q=u[i],df=df,gamma=gamma))
            return np.array(result)
        else:
            return skewt.ppf(q=u,df=df,gamma=gamma)

