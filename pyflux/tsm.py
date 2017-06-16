from math import exp, sqrt, log, tanh
import copy
import sys
import warnings
warnings.filterwarnings('ignore') # here to suppress nan-slice error warnings through optimization; not tackling root of problem...
if sys.version_info < (3,):
    range = xrange

import numpy as np
from scipy import optimize
import numdifftools as nd
import pandas as pd

from .covariances import acf
from .families import Normal
from .inference import BBVI, BBVIM, MetropolisHastings, norm_post_sim
from .output import TablePrinter
from .tests import find_p_value
from .latent_variables import LatentVariable, LatentVariables
from .results import BBVIResults, MLEResults, LaplaceResults, MCMCResults

class TSM(object):
    """ TSM PARENT CLASS

    Contains general time series methods to be inherited by models.

    Parameters
    ----------
    model_type : str
        The type of model (e.g. 'ARIMA', 'GARCH')
    """

    def __init__(self,model_type):

        # Holding variables for model output
        self.model_type = model_type
        self.latent_variables = LatentVariables(self.model_type)

    def _categorize_model_output(self, z):
        if self.model_type in ['GAS','GASX','GASLLEV','GARCH','EGARCH','EGARCHM']:
            theta, Y, scores = self._model(z)
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASRank']:
            theta, Y, states = self._model(z)
            scores = None
            states_var = None
            X_names = None       
        elif self.model_type in ['Skellam']:
            theta_1, theta_2, Y_1, Y_2 = self._model(z)
            Y = Y_1 - Y_2
            theta = np.exp(theta_1) - np.exp(theta_2)
            scores = None
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['DoubleGASRank']:
            theta_1, theta_2, Y_1, Y_2, _, _ = self._model(z)
            Y = Y_1 - Y_2
            theta = theta_1
            scores = None
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASLLT']:
            theta, mu_t, Y, scores = self._model(z)
            states = np.array([theta, mu_t])
            states_var = None
            X_names = None
        elif self.model_type in ['LMEGARCH']:
            theta, _, Y, scores = self._model(z)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['SEGARCH','SEGARCHM']:
            theta, Y, scores, y_theta = self._model(z)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['EGARCHMReg']:
            theta, Y, scores, _ = self._model(z)
            states = None
            states_var = None
            X_names = None           
        elif self.model_type in ['GASReg']:
            theta, Y, scores, states = self._model(z)
            states_var = None
            X_names = self.X_names
        elif self.model_type in ['LLEV','LLT','DynReg','DAR']:
            Y = self.data
            scores = None
            states, states_var = self.smoothed_state(self.data,z)
            theta = states[0][:-1]
            X_names = None  
        elif self.model_type in ['GPNARX','GPR','GP']:
            Y = self.data*self._norm_std + self._norm_mean
            scores = None
            theta = self.expected_values(z)*self._norm_std + self._norm_mean
            X_names = None  
            states = None   
            states_var = None
        else:
            theta, Y = self._model(z)
            scores = None
            states = None
            states_var = None
            X_names = None

        return theta, Y, scores, states, states_var, X_names

    def _bbvi_fit(self, posterior, optimizer='RMSProp', iterations=1000, 
        map_start=True, batch_size=12, mini_batch=None, learning_rate=0.001, 
        record_elbo=False, quiet_progress=False, **kwargs):
        """ Performs Black Box Variational Inference

        Parameters
        ----------
        posterior : method
            Hands bbvi_fit a posterior object

        optimizer : string
            Stochastic optimizer: one of RMSProp or ADAM.

        iterations: int
            How many iterations for BBVI

        map_start : boolean
            Whether to start values from a MAP estimate (if False, uses default starting values)

        Returns
        ----------
        BBVIResults object
        """

        # Starting values
        phi = self.latent_variables.get_z_starting_values()
        phi = kwargs.get('start',phi).copy() # If user supplied

        if self.model_type not in ['GPNARX','GPR','GP','GASRank'] and map_start is True and mini_batch is None:
            p = optimize.minimize(posterior, phi, method='L-BFGS-B') # PML starting values
            start_loc = 0.8*p.x + 0.2*phi
        else:
            start_loc = phi
        start_ses = None

        # Starting values for approximate distribution
        for i in range(len(self.latent_variables.z_list)):
            approx_dist = self.latent_variables.z_list[i].q
            if isinstance(approx_dist, Normal):
                if start_ses is None:
                    self.latent_variables.z_list[i].q.mu0 = start_loc[i]
                    self.latent_variables.z_list[i].q.sigma0 = np.exp(-3.0)
                else:
                    self.latent_variables.z_list[i].q.mu0 = start_loc[i]
                    self.latent_variables.z_list[i].q.sigma0 = start_ses[i]

        q_list = [k.q for k in self.latent_variables.z_list]

        if mini_batch is None:
            bbvi_obj = BBVI(posterior, q_list, batch_size, optimizer, iterations, learning_rate, record_elbo, quiet_progress)
        else:
            bbvi_obj = BBVIM(posterior, self.neg_logposterior, q_list, mini_batch, optimizer, iterations, learning_rate, mini_batch, record_elbo, quiet_progress)
        
        q, q_z, q_ses, elbo_records = bbvi_obj.run()
        self.latent_variables.set_z_values(q_z,'BBVI',np.exp(q_ses),None)

        for k in range(len(self.latent_variables.z_list)):
            self.latent_variables.z_list[k].q = q[k]

        self.latent_variables.estimation_method = 'BBVI'

        theta, Y, scores, states, states_var, X_names = self._categorize_model_output(q_z)

        # Change this in future
        try:
            latent_variables_store = self.latent_variables.copy()
        except:
            latent_variables_store = self.latent_variables

        return BBVIResults(data_name=self.data_name, X_names=X_names, model_name=self.model_name,
            model_type=self.model_type, latent_variables=latent_variables_store, data=Y, index=self.index,
            multivariate_model=self.multivariate_model, objective_object=self.neg_logposterior, 
            method='BBVI', ses=q_ses, signal=theta, scores=scores, elbo_records=elbo_records,
            z_hide=self._z_hide, max_lag=self.max_lag, states=states, states_var=states_var)

    def _laplace_fit(self,obj_type):
        """ Performs a Laplace approximation to the posterior

        Parameters
        ----------
        obj_type : method
            Whether a likelihood or a posterior

        Returns
        ----------
        None (plots posterior)
        """

        # Get Mode and Inverse Hessian information
        y = self.fit(method='PML',printer=False)

        if y.ihessian is None:
            raise Exception("No Hessian information - Laplace approximation cannot be performed")
        else:

            self.latent_variables.estimation_method = 'Laplace'

            theta, Y, scores, states, states_var, X_names = self._categorize_model_output(self.latent_variables.get_z_values())

            # Change this in future
            try:
                latent_variables_store = self.latent_variables.copy()
            except:
                latent_variables_store = self.latent_variables

            return LaplaceResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
                model_type=self.model_type, latent_variables=latent_variables_store,data=Y,index=self.index,
                multivariate_model=self.multivariate_model,objective_object=obj_type, 
                method='Laplace',ihessian=y.ihessian,signal=theta,scores=scores,
                z_hide=self._z_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def _mcmc_fit(self, scale=1.0, nsims=10000, printer=True, method="M-H", 
        cov_matrix=None, map_start=True, quiet_progress=False, **kwargs):
        """ Performs random walk Metropolis-Hastings

        Parameters
        ----------
        scale : float
            Default starting scale

        nsims : int
            Number of simulations

        printer : Boolean
            Whether to print results or not

        method : str
            What type of MCMC

        cov_matrix: None or np.array
            Can optionally provide a covariance matrix for M-H.
        """
        scale = 2.38/np.sqrt(self.z_no)

        # Get Mode and Inverse Hessian information
        if self.model_type in ['GPNARX', 'GPR', 'GP'] or map_start is True:
            y = self.fit(method='PML', printer=False)
            starting_values = y.z.get_z_values()

            # TODO: Bad use of a try/except - remove in future
            try:
                ses = np.abs(np.diag(y.ihessian))
                if len(ses[np.isnan(ses)]) != 0:
                    ses = np.ones(ses.shape[0])
                cov_matrix = np.zeros((len(ses), len(ses)))
                np.fill_diagonal(cov_matrix, ses)
            except:
                pass
        else:
            starting_values = self.latent_variables.get_z_starting_values()

        if method == "M-H":
            sampler = MetropolisHastings(self.neg_logposterior, scale, nsims, starting_values, 
                cov_matrix=cov_matrix, model_object=None, quiet_progress=quiet_progress)
            chain, mean_est, median_est, upper_95_est, lower_95_est = sampler.sample()
        else:
            raise Exception("Method not recognized!")

        if len(self.latent_variables.z_list) == 1:
            self.latent_variables.set_z_values(mean_est,'M-H',None,chain)
            mean_est = self.latent_variables.z_list[0].prior.transform(mean_est)
            median_est = self.latent_variables.z_list[0].prior.transform(median_est)
            upper_95_est = self.latent_variables.z_list[0].prior.transform(upper_95_est)
            lower_95_est = self.latent_variables.z_list[0].prior.transform(lower_95_est)        

        else:
            self.latent_variables.set_z_values(mean_est, 'M-H', None, chain)

            for k in range(len(chain)):
                mean_est[k] = self.latent_variables.z_list[k].prior.transform(mean_est[k])
                median_est[k] = self.latent_variables.z_list[k].prior.transform(median_est[k])
                upper_95_est[k] = self.latent_variables.z_list[k].prior.transform(upper_95_est[k])
                lower_95_est[k] = self.latent_variables.z_list[k].prior.transform(lower_95_est[k])        

        self.latent_variables.estimation_method = 'M-H'

        theta, Y, scores, states, states_var, X_names = self._categorize_model_output(mean_est)
    
        # Change this in future
        try:
            latent_variables_store = self.latent_variables.copy()
        except:
            latent_variables_store = self.latent_variables

        return MCMCResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, latent_variables=latent_variables_store,data=Y,index=self.index,
            multivariate_model=self.multivariate_model,objective_object=self.neg_logposterior, 
            method='Metropolis Hastings',samples=chain,mean_est=mean_est,median_est=median_est,lower_95_est=lower_95_est,
            upper_95_est=upper_95_est,signal=theta,scores=scores, z_hide=self._z_hide,max_lag=self.max_lag,
            states=states,states_var=states_var)

    def _ols_fit(self):
        """ Performs OLS

        Returns
        ----------
        None (stores latent variables)
        """

        # TO DO - A lot of things are VAR specific here; might need to refactor in future, or just move to VAR script

        method = 'OLS'
        self.use_ols_covariance = True
        
        res_z = self._create_B_direct().flatten()
        z = res_z.copy()
        cov = self.ols_covariance()

        # Inelegant - needs refactoring
        for i in range(self.ylen):
            for k in range(self.ylen):
                if i == k or i > k:
                    z = np.append(z,self.latent_variables.z_list[-1].prior.itransform(cov[i,k]))

        ihessian = self.estimator_cov('OLS')
        res_ses = np.power(np.abs(np.diag(ihessian)),0.5)
        ses = np.append(res_ses,np.ones([z.shape[0]-res_z.shape[0]]))
        self.latent_variables.set_z_values(z,method,ses,None)

        self.latent_variables.estimation_method = 'OLS'

        theta, Y, scores, states, states_var, X_names = self._categorize_model_output(z)

        # Change this in future
        try:
            latent_variables_store = self.latent_variables.copy()
        except:
            latent_variables_store = self.latent_variables

        return MLEResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, latent_variables=latent_variables_store,results=None,data=Y, index=self.index,
            multivariate_model=self.multivariate_model,objective_object=self.neg_loglik, 
            method=method,ihessian=ihessian,signal=theta,scores=scores,
            z_hide=self._z_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def _optimize_fit(self, obj_type=None, **kwargs):
        """
        This function fits models using Maximum Likelihood or Penalized Maximum Likelihood
        """

        preopt_search = kwargs.get('preopt_search', True) # If user supplied

        if obj_type == self.neg_loglik:
            method = 'MLE'
        else:
            method = 'PML'

        # Starting values - check to see if model has preoptimize method, if not, simply use default starting values
        if preopt_search is True:
            try:
                phi = self._preoptimize_model(self.latent_variables.get_z_starting_values(), method)
                preoptimized = True
            except:
                phi = self.latent_variables.get_z_starting_values()
                preoptimized = False
        else:
            preoptimized = False
            phi = self.latent_variables.get_z_starting_values()

        phi = kwargs.get('start',phi).copy() # If user supplied

        # Optimize using L-BFGS-B
        p = optimize.minimize(obj_type, phi, method='L-BFGS-B', options={'gtol': 1e-8})
        if preoptimized is True:
            p2 = optimize.minimize(obj_type, self.latent_variables.get_z_starting_values(), method='L-BFGS-B', 
                options={'gtol': 1e-8})
            if self.neg_loglik(p2.x) < self.neg_loglik(p.x):
                p = p2

        theta, Y, scores, states, states_var, X_names = self._categorize_model_output(p.x)

        # Check that matrix is non-singular; act accordingly
        try:
            ihessian = np.linalg.inv(nd.Hessian(obj_type)(p.x))
            ses = np.power(np.abs(np.diag(ihessian)),0.5)
            self.latent_variables.set_z_values(p.x,method,ses,None)

        except:
            ihessian = None
            ses = None
            self.latent_variables.set_z_values(p.x,method,None,None)

        self.latent_variables.estimation_method = method

        # Change this in future
        try:
            latent_variables_store = self.latent_variables.copy()
        except:
            latent_variables_store = self.latent_variables

        return MLEResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
                model_type=self.model_type, latent_variables=latent_variables_store,results=p,data=Y, index=self.index,
                multivariate_model=self.multivariate_model,objective_object=obj_type, 
                method=method,ihessian=ihessian,signal=theta,scores=scores,
                z_hide=self._z_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def fit(self, method=None, **kwargs):
        """ Fits a model

        Parameters
        ----------
        method : str
            A fitting method (e.g 'MLE'). Defaults to model specific default method.

        Returns
        ----------
        None (stores fit information)
        """

        cov_matrix = kwargs.get('cov_matrix', None)
        iterations = kwargs.get('iterations', 1000)
        nsims = kwargs.get('nsims', 10000)
        optimizer = kwargs.get('optimizer', 'RMSProp')
        batch_size = kwargs.get('batch_size', 12)
        mini_batch = kwargs.get('mini_batch', None)
        map_start = kwargs.get('map_start', True)
        learning_rate = kwargs.get('learning_rate', 0.001)
        record_elbo = kwargs.get('record_elbo', None)
        quiet_progress = kwargs.get('quiet_progress', False)

        if method is None:
            method = self.default_method
        elif method not in self.supported_methods:
            raise ValueError("Method not supported!")

        if method == 'MLE':
            return self._optimize_fit(self.neg_loglik, **kwargs)
        elif method == 'PML':
            return self._optimize_fit(self.neg_logposterior, **kwargs)   
        elif method == 'M-H':
            return self._mcmc_fit(nsims=nsims, method=method, cov_matrix=cov_matrix,
                map_start=map_start, quiet_progress=quiet_progress)
        elif method == "Laplace":
            return self._laplace_fit(self.neg_logposterior) 
        elif method == "BBVI":
            if mini_batch is None:
                posterior = self.neg_logposterior
            else:
                posterior = self.mb_neg_logposterior
            return self._bbvi_fit(posterior, optimizer=optimizer, iterations=iterations,
                batch_size=batch_size, mini_batch=mini_batch, map_start=map_start, 
                learning_rate=learning_rate, record_elbo=record_elbo, quiet_progress=quiet_progress)
        elif method == "OLS":
            return self._ols_fit()          

    def neg_logposterior(self, beta):
        """ Returns negative log posterior

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        Negative log posterior
        """

        post = self.neg_loglik(beta)
        for k in range(0,self.z_no):
            post += -self.latent_variables.z_list[k].prior.logpdf(beta[k])
        return post

    def mb_neg_logposterior(self, beta, mini_batch):
        """ Returns negative log posterior

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        mini_batch : int
            Batch size for the data

        Returns
        ----------
        Negative log posterior
        """

        post = (self.data.shape[0]/mini_batch)*self.mb_neg_loglik(beta, mini_batch)
        for k in range(0,self.z_no):
            post += -self.latent_variables.z_list[k].prior.logpdf(beta[k])
        return post

    def multivariate_neg_logposterior(self,beta):
        """ Returns negative log posterior, for a model with a covariance matrix 

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent_variables

        Returns
        ----------
        Negative log posterior
        """

        post = self.neg_loglik(beta)
        for k in range(0,self.z_no):
            if self.latent_variables.z_list[k].prior.covariance_prior is True:
                post += -self.latent_variables.z_list[k].prior.logpdf(self.custom_covariance(beta))
                break
            else:
                post += -self.latent_variables.z_list[k].prior.logpdf(beta[k])
        return post

    def shift_dates(self,h):
        """ Auxiliary function for creating dates for forecasts

        Parameters
        ----------
        h : int
            How many steps to forecast

        Returns
        ----------
        A transformed date_index object
        """

        date_index = copy.deepcopy(self.index)
        date_index = date_index[self.max_lag:len(date_index)]

        if self.is_pandas is True:

            if isinstance(date_index, pd.core.indexes.datetimes.DatetimeIndex):

                if pd.infer_freq(date_index) in ['H', 'M', 'S']:

                    for t in range(h):
                        date_index += pd.DateOffset((date_index[len(date_index)-1] - date_index[len(date_index)-2]).seconds)

                else: # Assume higher frequency (configured for days)

                    for t in range(h):
                        date_index += pd.DateOffset((date_index[len(date_index)-1] - date_index[len(date_index)-2]).days)

            elif isinstance(date_index, pd.core.indexes.numeric.Int64Index):

                for i in range(h):
                    new_value = date_index.values[len(date_index.values)-1] + (date_index.values[len(date_index.values)-1] - date_index.values[len(date_index.values)-2])
                    date_index = pd.Int64Index(np.append(date_index.values,new_value))

        else:

            for t in range(h):
                date_index.append(date_index[len(date_index)-1]+1)

        return date_index   

    def transform_z(self):
        """ Transforms latent variables to actual scale by applying link function

        Returns
        ----------
        Transformed latent variables 
        """
        return self.latent_variables.get_z_values(transformed=True)

    def transform_parameters(self):
        """ Frequentist notation for transform_latent_variables (maybe remove in future)

        Returns
        ----------
        Transformed latent variables 
        """
        return self.transformed_z()

    def plot_z(self, indices=None,figsize=(15,5),**kwargs):
        """ Plots latent variables by calling latent parameters object

        Returns
        ----------
        Pretty plot 
        """
        self.latent_variables.plot_z(indices=indices,figsize=figsize,**kwargs)

    def plot_parameters(self, indices=None,figsize=(15,5),**kwargs):
        """ Frequentist notation for plot_z (maybe remove in future)

        Returns
        ----------
        Pretty plot
        """
        self.plot_z(indices,figsize,**kwargs)

    def adjust_prior(self, index, prior):
        """ Adjusts priors for the latent variables

        Parameters
        ----------
        index : int or list[int]
            Which latent variable index/indices to be altered

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        Returns
        ----------
        None (changes priors in LatentVariables object)
        """
        self.latent_variables.adjust_prior(index=index, prior=prior)

    def draw_latent_variables(self, nsims=5000):
        """ Draws latent variables from the model (for Bayesian inference)

        Parameters
        ----------
        nsims : int
            How many draws to take

        Returns
        ----------
        - np.ndarray of draws
        """
        if self.latent_variables.estimation_method is None:
            raise Exception("No latent variables estimated!")
        elif self.latent_variables.estimation_method == 'BBVI':
            return np.array([i.q.draw_variable_local(size=nsims) for i in self.latent_variables.z_list])
        elif self.latent_variables.estimation_method == "M-H":
            chain = np.array([self.latent_variables.z_list[i].sample for i in range(len(self.latent_variables.z_list))])
            return chain[:,np.random.choice(chain.shape[1], nsims)]
        else:
            raise Exception("No latent variables estimated through Bayesian inference")