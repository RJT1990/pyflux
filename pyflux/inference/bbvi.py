import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as ss

from .stoch_optim import RMSProp, ADAM

class BBVI(object):
    """
    Black Box Variational Inference
    Parameters
    ----------
    neg_posterior : function
        posterior function
    q : List
        list holding distribution objects
    sims : int
        Number of Monte Carlo sims for the gradient
    step : float
        Step size for RMSProp
    iterations: int
        How many iterations to run
    """

    def __init__(self,neg_posterior,q,sims,optimizer='RMSProp',iterations=1000):
        self.neg_posterior = neg_posterior
        self.q = q
        self.sims = sims
        self.iterations = iterations
        self.approx_param_no = np.array([i.param_no for i in self.q])
        self.optimizer = optimizer
        self.printer = True

    def change_parameters(self,params):
        """
        Utility function for changing the approximate distribution parameters
        """
        no_of_params = 0
        for core_param in range(len(self.q)):
            for approx_param in range(self.q[core_param].param_no):
                self.q[core_param].change_param(approx_param,params[no_of_params])
                no_of_params += 1

    def create_normal_logq(self,z):
        """
        Create logq components for mean-field normal family (the entropy estimate)
        """
        means, scale = self.get_means_and_scales()
        return ss.norm.logpdf(z,loc=means,scale=scale).sum()

    def current_parameters(self):
        """
        Obtains an array with the current parameters
        """
        current = []
        for core_param in range(len(self.q)):
            for approx_param in range(self.q[core_param].param_no):
                current.append(self.q[core_param].return_param(approx_param))
        return np.array(current)

    def cv_gradient(self,z):
        """
        The control variate augmented Monte Carlo gradient estimate
        """
        gradient = np.zeros(np.sum(self.approx_param_no))
        alpha0 = np.zeros(np.sum(self.approx_param_no))  
        z_t = z.T      
        log_q = self.normal_log_q(z.T)
        log_p = self.log_p(z.T)
        grad_log_q = self.grad_log_q(z)
        gradient = grad_log_q*(log_p-log_q)

        for lambda_i in range(np.sum(self.approx_param_no)):
            alpha0[lambda_i] = np.cov(grad_log_q[lambda_i],gradient[lambda_i])[0][1]             

        vectorized = gradient - ((alpha0/np.var(grad_log_q,axis=1))*grad_log_q.T).T

        return np.mean(vectorized,axis=1)

    def cv_gradient_initial(self,z):
        """
        The control variate augmented Monte Carlo gradient estimate
        """
        gradient = np.zeros(np.sum(self.approx_param_no))
        alpha0 = np.zeros(np.sum(self.approx_param_no))  
        z_t = z.T      
        log_q = self.normal_log_q_initial(z.T)
        log_p = self.log_p(z.T)
        grad_log_q = self.grad_log_q(z)
        gradient = grad_log_q*(log_p-log_q)

        for lambda_i in range(np.sum(self.approx_param_no)):
            alpha0[lambda_i] = np.cov(grad_log_q[lambda_i],gradient[lambda_i])[0][1]             

        vectorized = gradient - ((alpha0/np.var(grad_log_q,axis=1))*grad_log_q.T).T

        return np.mean(vectorized,axis=1)

    def draw_normal(self):
        """
        Draw parameters from a mean-field normal family
        """
        means, scale = self.get_means_and_scales()
        return np.random.normal(means,scale,size=[self.sims,means.shape[0]]).T

    def draw_normal_initial(self):
        """
        Draw parameters from a mean-field normal family
        """
        means, scale = self.get_means_and_scales_from_q()
        return np.random.normal(means,scale,size=[self.sims,means.shape[0]]).T

    def draw_variables(self):
        """
        Draw parameters from the approximating distributions
        """        
        z = self.q[0].sim(self.sims)
        for i in range(1,len(self.q)):
            z = np.vstack((z,self.q[i].sim(self.sims)))
        return z

    def get_means_and_scales_from_q(self):
        """
        Gets the mean and scales for normal approximating parameters
        """
        means = np.zeros(len(self.q))
        scale = np.zeros(len(self.q))
        for i in range(len(self.q)):
            means[i] = self.q[i].loc
            scale[i] = self.q[i].scale  
        return means, np.exp(scale)

    def get_means_and_scales(self):
        """
        Gets the mean and scales for normal approximating parameters
        """
        return self.optim.parameters[::2], np.exp(self.optim.parameters[1::2])

    def grad_log_q(self,z):
        """
        The gradients of the approximating distributions
        """        
        param_count = 0
        grad = np.zeros((np.sum(self.approx_param_no),self.sims))
        for core_param in range(len(self.q)):
            for approx_param in range(self.q[core_param].param_no):
                grad[param_count] = self.q[core_param].score(z[core_param],approx_param)        
                param_count += 1
        return grad

    def log_p(self,z):
        """
        The unnormalized log posterior components (the quantity we want to approximate)
        """
        return np.array([-self.neg_posterior(i) for i in z])

    def normal_log_q(self,z):
        """
        The mean-field normal log posterior components (the quantity we want to approximate)
        """
        means, scale = self.get_means_and_scales()
        return (ss.norm.logpdf(z,loc=means,scale=scale)).sum(axis=1)

    def normal_log_q_initial(self,z):
        """
        The mean-field normal log posterior components (the quantity we want to approximate)
        """
        means, scale = self.get_means_and_scales_from_q()
        return (ss.norm.logpdf(z,loc=means,scale=scale)).sum(axis=1)

    def print_progress(self,i,current_params):
        """
        Prints the current ELBO at every decile of total iterations
        """
        for split in range(1,11):
            if i == (round(self.iterations/10*split)-1):
                print(str(split) + "0% done : ELBO is " + str(-self.neg_posterior(current_params)-self.create_normal_logq(current_params)))

    def run(self):
        """
        The core BBVI routine - draws Monte Carlo gradients and uses a stochastic optimizer.
        """

        # Initialization assumptions
        z = self.draw_normal_initial()
        gradient = self.cv_gradient_initial(z)
        gradient[np.isnan(gradient)] = 0
        variance = np.power(gradient,2)       
        final_parameters = self.current_parameters()
        final_samples = 1

        # Create optimizer
        if self.optimizer == 'ADAM':
            self.optim = ADAM(final_parameters,variance,0.001,0.9,0.999)
        elif self.optimizer == 'RMSProp':
            self.optim = RMSProp(final_parameters,variance,0.001,0.99)

        for i in range(self.iterations):
            gradient = self.cv_gradient(self.draw_normal())
            gradient[np.isnan(gradient)] = 0
            self.change_parameters(self.optim.update(gradient))

            if self.printer is True:
                self.print_progress(i,self.optim.parameters[::2])

            # Construct final parameters using final 10% of samples
            if i > self.iterations-round(self.iterations/10):
                final_samples += 1
                final_parameters = final_parameters+self.optim.parameters

        final_parameters = final_parameters/float(final_samples)
        final_means = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2==0])
        final_ses = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2!=0])
        if self.printer is True:
            print("")
            print("Final model ELBO is " + str(-self.neg_posterior(final_means)-self.create_normal_logq(final_means)))
        return self.q, final_means, final_ses

    def run_and_store(self):
        """
        The core BBVI routine - draws Monte Carlo gradients and uses a stochastic optimizer. 
        Stores rgw history of updates for the benefit of a pretty animation.
        """
        # Initialization assumptions
        z = self.draw_normal_initial()
        gradient = self.cv_gradient_initial(z)
        gradient[np.isnan(gradient)] = 0
        variance = np.power(gradient,2)       
        final_parameters = self.current_parameters()
        final_samples = 1

        # Create optimizer
        if self.optimizer == 'ADAM':
            self.optim = ADAM(final_parameters,variance,0.001,0.9,0.999)
        elif self.optimizer == 'RMSProp':
            self.optim = RMSProp(final_parameters,variance,0.001,0.99)

        # Stored updates
        stored_means = np.zeros((self.iterations,len(final_parameters)/2))
        stored_predictive_likelihood = np.zeros(self.iterations)

        for i in range(self.iterations):
            gradient = self.cv_gradient(self.draw_normal())
            gradient[np.isnan(gradient)] = 0
            new_parameters = self.optim.update(gradient)
            self.change_parameters(new_parameters)

            stored_means[i] = self.optim.parameters[::2]
            stored_predictive_likelihood[i] = self.neg_posterior(stored_means[i])

            if self.printer is True:
                self.print_progress(i,self.optim.parameters[::2])

            # Construct final parameters using final 10% of samples
            if i > self.iterations-round(self.iterations/10):
                final_samples += 1
                final_parameters = final_parameters+self.optim.parameters

        final_parameters = final_parameters/float(final_samples)
        final_means = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2==0])
        final_ses = np.array([final_parameters[el] for el in range(len(final_parameters)) if el%2!=0])

        if self.printer is True:
            print("")
            print("Final model ELBO is " + str(-self.neg_posterior(final_means)-self.create_normal_logq(final_means)))
        return self.q, final_means, final_ses, stored_means, stored_predictive_likelihood

class CBBVI(BBVI):

    def __init__(self,neg_posterior,log_p_blanket,q,sims,optimizer='RMSProp',iterations=300000):
        super(CBBVI,self).__init__(neg_posterior,q,sims,optimizer,iterations)
        self.log_p_blanket = log_p_blanket

    def log_p(self,z):
        """
        The unnormalized log posterior components (the quantity we want to approximate)
        RAO-BLACKWELLIZED!
        """        
        return np.array([self.log_p_blanket(i) for i in z])

    def normal_log_q(self,z):
        """
        The unnormalized log posterior components for mean-field normal family (the quantity we want to approximate)
        RAO-BLACKWELLIZED!
        """             
        means, scale = self.get_means_and_scales()
        return ss.norm.logpdf(z,loc=means,scale=scale)

    def normal_log_q_initial(self,z):
        """
        The unnormalized log posterior components for mean-field normal family (the quantity we want to approximate)
        RAO-BLACKWELLIZED!
        """             
        means, scale = self.get_means_and_scales_from_q()
        return ss.norm.logpdf(z,loc=means,scale=scale)

    def cv_gradient(self,z):
        """
        The control variate augmented Monte Carlo gradient estimate
        RAO-BLACKWELLIZED!
        """        
        alpha0 = np.zeros(np.sum(self.approx_param_no))        
        z_t = np.transpose(z)
        log_q = self.normal_log_q(z_t)
        log_p = self.log_p(z_t)
        grad_log_q = self.grad_log_q(z)
        gradient = grad_log_q*np.repeat((log_p - log_q).T,2,axis=0)

        for lambda_i in range(np.sum(self.approx_param_no)):
            alpha0[lambda_i] = np.cov(grad_log_q[lambda_i],gradient[lambda_i])[0][1]              
        
        vectorized = gradient - ((alpha0/np.var(grad_log_q,axis=1))*grad_log_q.T).T

        return np.mean(vectorized,axis=1)

    def cv_gradient_initial(self,z):
        """
        The control variate augmented Monte Carlo gradient estimate
        RAO-BLACKWELLIZED!
        """        
        alpha0 = np.zeros(np.sum(self.approx_param_no))        
        z_t = np.transpose(z)
        log_q = self.normal_log_q_initial(z_t)
        log_p = self.log_p(z_t)
        grad_log_q = self.grad_log_q(z)
        gradient = grad_log_q*np.repeat((log_p - log_q).T,2,axis=0)

        for lambda_i in range(np.sum(self.approx_param_no)):
            alpha0[lambda_i] = np.cov(grad_log_q[lambda_i],gradient[lambda_i])[0][1]              
        
        vectorized = gradient - ((alpha0/np.var(grad_log_q,axis=1))*grad_log_q.T).T

        return np.mean(vectorized,axis=1)