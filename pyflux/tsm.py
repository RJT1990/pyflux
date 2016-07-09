from math import exp, sqrt, log, tanh
import copy
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numdifftools as nd
import pandas as pd

from .covariances import acf
from .inference import BBVI, MetropolisHastings, norm_post_sim, Normal, InverseGamma, Uniform
from .output import TablePrinter
from .tests import find_p_value
from .distributions import q_Normal
from .parameter import Parameter, Parameters
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
        self.parameters = Parameters(self.model_type)

    def _bbvi_fit(self,posterior,optimizer='RMSProp',iterations=1000,**kwargs):
        """ Performs Black Box Variational Inference

        Parameters
        ----------
        posterior : method
            Hands bbvi_fit a posterior object

        optimizer : string
            Stochastic optimizer: one of RMSProp or ADAM.

        iterations: int
            How many iterations for BBVI

        Returns
        ----------
        BBVIResults object
        """

        # Starting parameters
        phi = self.parameters.get_parameter_starting_values()
        phi = kwargs.get('start',phi).copy() # If user supplied
        batch_size = kwargs.get('batch_size',12) # If user supplied
        p = optimize.minimize(posterior,phi,method='L-BFGS-B') # PML starting values
        start_loc = 0.8*p.x + 0.2*phi
        start_ses = None

        # Starting values for approximate distribution
        for i in range(len(self.parameters.parameter_list)):
            approx_dist = self.parameters.parameter_list[i].q
            if isinstance(approx_dist, q_Normal):
                if start_ses is None:
                    self.parameters.parameter_list[i].q.loc = start_loc[i]
                    self.parameters.parameter_list[i].q.scale = -3.0
                else:
                    self.parameters.parameter_list[i].q.loc = start_loc[i]
                    self.parameters.parameter_list[i].q.scale = start_ses[i]

        q_list = [k.q for k in self.parameters.parameter_list]
        
        bbvi_obj = BBVI(posterior,q_list,batch_size,optimizer,iterations)
        q, q_params, q_ses = bbvi_obj.run()
        self.parameters.set_parameter_values(q_params,'BBVI',np.exp(q_ses),None)

        for k in range(len(self.parameters.parameter_list)):
            self.parameters.parameter_list[k].q = q[k]

        if self.model_type in ['GAS','GASX','GASLLEV','GARCH','EGARCH','EGARCHM']:
            theta, Y, scores = self._model(q_params)
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASLLT']:
            theta, mu_t, Y, scores = self._model(q_params)
            states = np.array([theta, mu_t])
            states_var = None
            X_names = None
        elif self.model_type in ['LMEGARCH']:
            theta, _, Y, scores = self._model(q_params)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['SEGARCH','SEGARCHM']:
            theta, Y, scores, y_theta = self._model(q_params)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['EGARCHMReg']:
            theta, Y, scores, _ = self._model(q_params)
            states = None
            states_var = None
            X_names = None           
        elif self.model_type in ['GASReg']:
            theta, Y, scores, states = self._model(q_params)
            states_var = None
            X_names = self.X_names
        elif self.model_type in ['LLEV','LLT','DynLin']:
            Y = self.data
            scores = None
            states, states_var = self.smoothed_state(self.data,q_params)
            theta = states[0][:-1]
            X_names = None  
        elif self.model_type in ['GPNARX','GPR','GP']:
            Y = self.data*self._norm_std + self._norm_mean
            scores = None
            theta = self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean
            X_names = None  
            states = None   
            states_var = None
        else:
            theta, Y = self._model(q_params)
            scores = None
            states = None
            states_var = None
            X_names = None

        # Change this in future
        try:
            parameter_store = self.parameters.copy()
        except:
            parameter_store = self.parameters

        return BBVIResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, parameters=parameter_store,data=Y, index=self.index,
            multivariate_model=self.multivariate_model,objective_object=posterior, 
            method='BBVI',ses=q_ses,signal=theta,scores=scores,
            param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=states_var)

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
            if self.model_type in ['GAS','GASX','GASLLEV','GARCH','EGARCH','EGARCHM']:
                theta, Y, scores = self._model(y.parameters.get_parameter_values())
                states = None
                states_var = None
                X_names = None
            elif self.model_type in ['GASLLT']:
                theta, mu_t, Y, scores = self._model(y.parameters.get_parameter_values())
                states = np.array([theta, mu_t])
                states_var = None
                X_names = None        
            elif self.model_type in ['LMEGARCH']:
                theta, _, Y, scores = self._model(y.parameters.get_parameter_values())
                states = None
                states_var = None
                X_names = None        
            elif self.model_type in ['SEGARCH','SEGARCHM']:
                theta, Y, scores, y_theta = self._model(y.parameters.get_parameter_values())
                states = None
                states_var = None
                X_names = None        
            elif self.model_type in ['EGARCHMReg']:
                theta, Y, scores, _ = self._model(y.parameters.get_parameter_values())
                states = None
                states_var = None
                X_names = None                    
            elif self.model_type in ['GASReg']:
                theta, Y, scores, states = self._model(y.parameters.get_parameter_values())
                states_var = None
                X_names = self.X_names
            elif self.model_type in ['LLEV','LLT','DynLin']:
                Y = self.data
                scores = None
                states, states_var = self.smoothed_state(self.data,y.parameters.get_parameter_values())
                theta = states[0][:-1]
                X_names = None  
            elif self.model_type in ['GPNARX','GPR','GP']:
                Y = self.data*self._norm_std + self._norm_mean
                scores = None
                theta = self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean
                X_names = None  
                states = None   
                states_var = None
            else:
                theta, Y = self._model(y.parameters.get_parameter_values())
                scores = None
                states = None
                states_var = None
                X_names = None

            # Change this in future
            try:
                parameter_store = y.parameters.copy()
            except:
                parameter_store = y.parameters

            return LaplaceResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
                model_type=self.model_type, parameters=parameter_store,data=Y,index=self.index,
                multivariate_model=self.multivariate_model,objective_object=obj_type, 
                method='Laplace',ihessian=y.ihessian,signal=theta,scores=scores,
                param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def _mcmc_fit(self,scale=(2.38/sqrt(1000)),nsims=100000,printer=True,method="M-H",cov_matrix=None,**kwargs):
        """ Performs MCMC 

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

        Returns
        ----------
        None (plots posteriors, stores parameters)
        """

        # Get Mode and Inverse Hessian information
        y = self.fit(method='PML',printer=False)

        if method == "M-H":
            sampler = MetropolisHastings(self.neg_logposterior,scale,nsims,y.parameters.get_parameter_values(),cov_matrix=cov_matrix,model_object=None)
            chain, mean_est, median_est, upper_95_est, lower_95_est = sampler.sample()
        else:
            raise Exception("Method not recognized!")

        for k in range(len(chain)):
            chain[k] = self.parameters.parameter_list[k].prior.transform(chain[k])
            mean_est[k] = self.parameters.parameter_list[k].prior.transform(mean_est[k])
            median_est[k] = self.parameters.parameter_list[k].prior.transform(median_est[k])
            upper_95_est[k] = self.parameters.parameter_list[k].prior.transform(upper_95_est[k])
            lower_95_est[k] = self.parameters.parameter_list[k].prior.transform(lower_95_est[k])        

        self.parameters.set_parameter_values(mean_est,'M-H',None,chain)

        if self.model_type in ['GAS','GASX','GARCH','GASLLEV','EGARCH','EGARCHM']:
            theta, Y, scores = self._model(mean_est)
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASLLT']:
            theta, mu_t, Y, scores = self._model(mean_est)
            states = np.array([theta, mu_t])
            states_var = None
            X_names = None    
        elif self.model_type in ['LMEGARCH']:
            theta, _, Y, scores = self._model(mean_est)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['SEGARCH','SEGARCHM']:
            theta, Y, scores, y_theta = self._model(mean_est)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['EGARCHMReg']:
            theta, Y, scores, _ = self._model(mean_est)
            states = None
            states_var = None
            X_names = None                
        elif self.model_type in ['GASReg']:
            theta, Y, scores, states = self._model(mean_est)
            states_var = None
            X_names = self.X_names
        elif self.model_type in ['LLEV','LLT','DynLin']:
            Y = self.data
            scores = None
            states, states_var = self.smoothed_state(self.data,mean_est)
            theta = states[0][:-1]
            X_names = None  
        elif self.model_type in ['GPNARX','GPR','GP']:
            Y = self.data*self._norm_std + self._norm_mean
            scores = None
            theta = self.expected_values(mean_est)*self._norm_std + self._norm_mean
            X_names = None  
            states = None   
            states_var = None
        else:
            theta, Y = self._model(mean_est)
            scores = None
            states = None
            states_var = None
            X_names = None
    
        # Change this in future
        try:
            parameter_store = self.parameters.copy()
        except:
            parameter_store = self.parameters

        return MCMCResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, parameters=parameter_store,data=Y,index=self.index,
            multivariate_model=self.multivariate_model,objective_object=self.neg_logposterior, 
            method='Metropolis Hastings',samples=chain,mean_est=mean_est,median_est=median_est,lower_95_est=lower_95_est,
            upper_95_est=upper_95_est,signal=theta,scores=scores, param_hide=self._param_hide,max_lag=self.max_lag,
            states=states,states_var=states_var)

    def _ols_fit(self):
        """ Performs OLS

        Returns
        ----------
        None (stores parameters)
        """

        # TO DO - A lot of things are VAR specific here; might need to refactor in future, or just move to VAR script

        method = 'OLS'

        res_params = self._create_B_direct().flatten()
        params = res_params.copy()
        cov = self.ols_covariance()

        # Inelegant - needs refactoring
        for i in range(self.ylen):
            for k in range(self.ylen):
                if i == k or i > k:
                    params = np.append(params,self.parameters.parameter_list[-1].prior.itransform(cov[i,k]))

        ihessian = self.estimator_cov('OLS')
        res_ses = np.power(np.abs(np.diag(ihessian)),0.5)
        ses = np.append(res_ses,np.ones([params.shape[0]-res_params.shape[0]]))
        self.parameters.set_parameter_values(params,method,ses,None)

        if self.model_type in ['GAS','GASX','GARCH','EGARCH','GASLLEV','EGARCHM']:
            theta, Y, scores = self._model(params)
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASLLT']:
            theta, mu_t, Y, scores = self._model(params)
            states = np.array([theta, mu_t])
            states_var = None
            X_names = None    
        elif self.model_type in ['LMEGARCH']:
            theta, _, Y, scores = self._model(params)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['SEGARCH','SEGARCHM']:
            theta, Y, scores, y_theta = self._model(params)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['EGARCHMReg']:
            theta, Y, scores, _ = self._model(params)
            states = None
            states_var = None
            X_names = None                
        elif self.model_type in ['GASReg']:
            theta, Y, scores, states = self._model(params)
            X_names = self.X_names
            states_var = None
        elif self.model_type in ['LLEV','LLT','DynLin']:
            Y = self.data
            scores = None
            states, states_var = self.smoothed_state(self.data,params)
            theta = states[0][:-1]
            X_names = None
        elif self.model_type in ['GPNARX','GPR','GP']:
            Y = self.data*self._norm_std + self._norm_mean
            scores = None
            theta = self.expected_values(params)*self._norm_std + self._norm_mean
            X_names = None  
            states = None   
            states_var = None
        else:
            theta, Y = self._model(params)
            scores = None
            states = None
            X_names = None
            states_var = None

        # Change this in future
        try:
            parameter_store = self.parameters.copy()
        except:
            parameter_store = self.parameters

        return MLEResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, parameters=parameter_store,results=None,data=Y, index=self.index,
            multivariate_model=self.multivariate_model,objective_object=self.neg_loglik, 
            method=method,ihessian=ihessian,signal=theta,scores=scores,
            param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def _optimize_fit(self,obj_type=None,**kwargs):

        if obj_type == self.neg_loglik:
            method = 'MLE'
        else:
            method = 'PML'

        # Starting parameters
        phi = self.parameters.get_parameter_starting_values()
        phi = kwargs.get('start',phi).copy() # If user supplied

        # Optimize using L-BFGS-B
        p = optimize.minimize(obj_type,phi,method='L-BFGS-B')

        # Model check
        if self.model_type in ['GAS','GASX','GARCH','EGARCH','GASLLEV','EGARCHM']:
            theta, Y, scores = self._model(p.x)
            states = None
            states_var = None
            X_names = None
        elif self.model_type in ['GASLLT']:
            theta, mu_t, Y, scores = self._model(p.x)
            states = np.array([theta, mu_t])
            states_var = None
            X_names = None
        elif self.model_type in ['LMEGARCH']:
            theta, _, Y, scores = self._model(p.x)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['SEGARCH','SEGARCHM']:
            theta, Y, scores, y_theta = self._model(p.x)
            states = None
            states_var = None
            X_names = None    
        elif self.model_type in ['EGARCHMReg']:
            theta, Y, scores, _ = self._model(p.x)
            states = None
            states_var = None
            X_names = None                
        elif self.model_type in ['GASReg']:
            theta, Y, scores, states = self._model(p.x)
            X_names = self.X_names
            states_var = None
        elif self.model_type in ['LLEV','LLT','DynLin']:
            Y = self.data
            scores = None
            states, states_var = self.smoothed_state(self.data,p.x)
            theta = states[0][:-1]
            X_names = None
        elif self.model_type in ['GPNARX','GPR','GP']:
            Y = self.data*self._norm_std + self._norm_mean
            scores = None
            theta = self.expected_values(self.parameters.get_parameter_values())*self._norm_std + self._norm_mean
            X_names = None  
            states = None   
            states_var = None
        else:
            theta, Y = self._model(p.x)
            scores = None
            states = None
            X_names = None
            states_var = None

        # Check that matrix is non-singular; act accordingly
        try:
            ihessian = np.linalg.inv(nd.Hessian(obj_type)(p.x))
            ses = np.power(np.abs(np.diag(ihessian)),0.5)
            self.parameters.set_parameter_values(p.x,method,ses,None)

            # Change this in future
            try:
                parameter_store = self.parameters.copy()
            except:
                parameter_store = self.parameters

            return MLEResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
                model_type=self.model_type, parameters=parameter_store,results=p,data=Y, index=self.index,
                multivariate_model=self.multivariate_model,objective_object=obj_type, 
                method=method,ihessian=ihessian,signal=theta,scores=scores,
                param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=states_var)
        except:
            self.parameters.set_parameter_values(p.x,method,None,None)
 
            # Change this in future
            try:
                parameter_store = self.parameters.copy()
            except:
                parameter_store = self.parameters

            return MLEResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
                model_type=self.model_type,parameters=parameter_store,results=p,data=Y, index=self.index,
                multivariate_model=self.multivariate_model,objective_object=obj_type, 
                method=method,ihessian=None,signal=theta,scores=scores,
                param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def fit(self,method=None,**kwargs):
        """ Fits a model

        Parameters
        ----------
        method : str
            A fitting method (e.g 'MLE'). Defaults to model specific default method.

        Returns
        ----------
        None (stores fit information)
        """

        cov_matrix = kwargs.get('cov_matrix',None)
        iterations = kwargs.get('iterations',1000)
        nsims = kwargs.get('nsims',10000)
        optimizer = kwargs.get('optimizer','RMSProp')
        batch_size = kwargs.get('batch_size',12)

        if method is None:
            method = self.default_method
        elif method not in self.supported_methods:
            raise ValueError("Method not supported!")

        if method == 'MLE':
            return self._optimize_fit(self.neg_loglik,**kwargs)
        elif method == 'PML':
            return self._optimize_fit(self.neg_logposterior,**kwargs)   
        elif method == 'M-H':
            return self._mcmc_fit(nsims=nsims,method=method,cov_matrix=cov_matrix)
        elif method == "Laplace":
            return self._laplace_fit(self.neg_logposterior) 
        elif method == "BBVI":
            return self._bbvi_fit(self.neg_logposterior,optimizer=optimizer,iterations=iterations,batch_size=batch_size)
        elif method == "OLS":
            return self._ols_fit()          

    def neg_logposterior(self,beta):
        """ Returns negative log posterior

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        Negative log posterior
        """

        post = self.neg_loglik(beta)
        for k in range(0,self.param_no):
            post += -self.parameters.parameter_list[k].prior.logpdf(beta[k])
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

            if isinstance(date_index,pd.tseries.index.DatetimeIndex):

                # Only configured for days - need to support smaller time intervals!
                for t in range(h):
                    date_index += pd.DateOffset((date_index[len(date_index)-1] - date_index[len(date_index)-2]).days)

            elif isinstance(date_index,pd.core.index.Int64Index):

                for i in range(h):
                    new_value = date_index.values[len(date_index.values)-1] + (date_index.values[len(date_index.values)-1] - date_index.values[len(date_index.values)-2])
                    date_index = pd.Int64Index(np.append(date_index.values,new_value))

        else:

            for t in range(h):
                date_index.append(date_index[len(date_index)-1]+1)

        return date_index   

    def transform_parameters(self):
        """ Transforms parameters to actual scale by applying link function

        Returns
        ----------
        Transformed parameters 
        """     
        return self.parameters.get_parameter_values(transformed=True)

    def plot_parameters(self,indices=None,figsize=(15,5),**kwargs):
        """ Plots parameters by calling parameter object

        Returns
        ----------
        Pretty plot
        """     

        self.parameters.plot_parameters(indices=indices,figsize=figsize,**kwargs)

    def adjust_prior(self,index,prior):
        """ Adjusts priors for the parameters

        Parameters
        ----------
        index : int or list[int]
            Which parameter index/indices to be altered

        prior : Prior object
            Which prior distribution? E.g. Normal(0,1)

        Returns
        ----------
        None (changes priors in Parameters object)
        """

        self.parameters.adjust_prior(index=index,prior=prior)