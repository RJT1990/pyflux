import numpy as np
import sys
if sys.version_info < (3,):
    range = xrange

from scipy.stats import multivariate_normal

from .metropolis_sampler import metropolis_sampler

class MetropolisHastings(object):
    """ RANDOM-WALK METROPOLIS-HASTINGS MCMC

    Parameters
    ----------
    posterior : function
        A posterior function

    scale : float
        The scale for the random walk

    nsims : int
        The number of iterations to perform

    initials : np.array
        Where to start the MCMC chain

    cov_matrix : np.array
        (optional) A covariance matrix for the random walk
    
    thinning : int
        By how much to thin the chains (2 means drop every other point)

    warm_up_period : boolean
        Whether to discard first half of the chain as 'warm-up'

    model_object : TSM object
        A model object (for use in SPDK sampling)

    quiet_progress : boolean
        Whether to print progress to console or stay quiet
    """

    def __init__(self, posterior, scale, nsims, initials, 
        cov_matrix=None, thinning=2, warm_up_period=True, model_object=None, quiet_progress=False):
        self.posterior = posterior
        self.scale = scale
        self.nsims = (1+warm_up_period)*nsims*thinning
        self.initials = initials
        self.param_no = self.initials.shape[0]
        self.phi = np.zeros([self.nsims, self.param_no])
        self.phi[0] = self.initials # point from which to start the Metropolis-Hasting algorithm
        self.quiet_progress = quiet_progress

        if cov_matrix is None:
            self.cov_matrix = np.identity(self.param_no) * np.abs(self.initials)
        else:
            self.cov_matrix = cov_matrix

        self.thinning = thinning
        self.warm_up_period = warm_up_period

        if model_object is not None:
            self.model = model_object

    @staticmethod
    def tune_scale(acceptance, scale):
        """ Tunes scale for M-H algorithm

        Parameters
        ----------
        acceptance : float
            The most recent acceptance rate

        scale : float
            The current scale parameter

        Returns
        ----------
        scale : float
            An adjusted scale parameter

        Notes
        ----------
        Ross : Initially did this by trial and error, then refined by looking at other
        implementations, so some credit here to PyMC3 which became a guideline for this.
        """     

        if acceptance > 0.8:
            scale *= 2.0
        elif acceptance <= 0.8 and acceptance > 0.4:
            scale *= 1.3            
        elif acceptance < 0.234 and acceptance > 0.1:
            scale *= (1/1.3)
        elif acceptance <= 0.1 and acceptance > 0.05:
            scale *= 0.4
        elif acceptance <= 0.05 and acceptance > 0.01:
            scale *= 0.2
        elif acceptance <= 0.01:
            scale *= 0.1
        return scale        

    def sample(self):
        """ Sample from M-H algorithm

        Returns
        ----------
        chain : np.array
            Chains for each parameter

        mean_est : np.array
            Mean values for each parameter

        median_est : np.array
            Median values for each parameter

        upper_95_est : np.array
            Upper 95% credibility interval for each parameter

        lower_95_est : np.array
            Lower 95% credibility interval for each parameter           
        """     

        acceptance = 1
        finish = 0

        while (acceptance < 0.234 or acceptance > 0.4) or finish == 0:

            # If acceptance is in range, proceed to sample, else continue tuning
            if not (acceptance < 0.234 or acceptance > 0.4):
                finish = 1
                if not self.quiet_progress:
                    print("")
                    print("Tuning complete! Now sampling.")
                sims_to_do = self.nsims
            else:
                sims_to_do = int(self.nsims/2) # For acceptance rate tuning

            # Holds data on acceptance rates and uniform random numbers
            a_rate = np.zeros([sims_to_do,1])
            crit = np.random.rand(sims_to_do,1)
            post = multivariate_normal(np.zeros(self.param_no), self.cov_matrix)
            rnums = post.rvs()*self.scale
            
            for k in range(1,sims_to_do): 
                rnums = np.vstack((rnums,post.rvs()*self.scale))

            self.phi, a_rate = metropolis_sampler(sims_to_do, self.phi, self.posterior, 
                a_rate, rnums, crit)

            acceptance = a_rate.sum()/a_rate.shape[0]
            self.scale = self.tune_scale(acceptance,self.scale)
            if not self.quiet_progress:
                print("Acceptance rate of Metropolis-Hastings is " + str(acceptance))

        # Remove warm-up and thin
        self.phi = self.phi[int(self.nsims/2):,:][::self.thinning,:]

        chain = np.array([self.phi[i][0] for i in range(0, self.phi.shape[0])])

        for m in range(1, self.param_no):
            chain = np.vstack((chain, [self.phi[i][m] for i in range(0,self.phi.shape[0])]))

        if self.param_no == 1:
            chain = np.array([chain])

        mean_est = np.array([np.mean(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
        median_est = np.array([np.median(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])])) for j in range(self.param_no)])
        upper_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]), 95) for j in range(self.param_no)])
        lower_95_est = np.array([np.percentile(np.array([self.phi[i][j] for i in range(0,self.phi.shape[0])]), 5) for j in range(self.param_no)])

        return chain, mean_est, median_est, upper_95_est, lower_95_est