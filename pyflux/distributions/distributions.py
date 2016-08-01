import scipy.stats as ss
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

class q_Normal(object):
    """
    Normal distribution class of variational approximation for BBVI
    """
    def __init__(self,loc,scale):
        self.loc = loc
        self.scale = scale
        self.param_no = 2

    def sim(self, size): 
        """ Simulate from the Normal distribution

        Parameters
        ----------
        size : int
            How many simulations to perform

        Returns
        ----------
        np.ndarray of Normal random variable
        """
        return ss.norm.rvs(loc=self.loc,scale=np.exp(self.scale),size=size)

    def loc_score(self,x):
        """ The gradient of the location latent variable mu

        Parameters
        ----------
        x : float
            A random variable

        Returns
        ----------
        The gradient of the location latent variable mu at x
        """
        return (x-self.loc)/(np.exp(self.scale)**2)

    def scale_score(self,x):
        """ The score of the scale, where scale = exp(x)

        Parameters
        ----------
        x : float
            A random variable

        Returns
        ----------
        The gradient of the scale latent variable at x
        """
        return np.exp(-2*self.scale)*(x-self.loc)**2 - 1 

    def score(self, x, index):
        """ Wrapper function for selecting appropriate score

        Parameters
        ----------
        x : float
            A random variable

        index : int
            0 or 1 depending on which latent variable

        Returns
        ----------
        The gradient of the scale latent variable at x
        """
        if index == 0:
            return self.loc_score(x)
        elif index == 1:
            return self.scale_score(x)

    def return_param(self, index):
        """ Wrapper function for selecting appropriate latent variable

        Parameters
        ----------
        index : int
            0 or 1 depending on which latent variable

        Returns
        ----------
        The appropriate indexed parameter
        """
        if index == 0:
            return self.loc
        elif index == 1:
            return self.scale

    def change_param(self, index, value):
        """ Wrapper function for changing latent variables

        Parameters
        ----------
        index : int
            0 or 1 depending on which latent variable

        value : float
            What to change the latent variable to
        """
        if index == 0:
            self.loc = value
        elif index == 1:
            self.scale = value

    def logpdf(self, x):
        """ Log PDF of Normal Distribution

        Parameters
        ----------
        x : float
            random variable

        Returns
        ----------
        - float: log PDF of the normal distribution
        """
        return ss.norm.logpdf(x,loc=self.loc,scale=np.exp(self.scale))

    def plot_pdf(self):
        """
        Plots the PDF of the Normal Distribution
        """
        x = np.linspace(self.loc-np.exp(self.scale)*3.5,self.loc+np.exp(self.scale)*3.5,100)
        plt.plot(x,mlab.normpdf(x,self.loc,np.exp(self.scale)))
        plt.show()
