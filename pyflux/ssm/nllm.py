import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import seaborn as sns

from .. import inference as ifr
from .. import distributions as dst
from .. import output as op
from .. import tsm as tsm
from .. import data_check as dc
from .. import covariances as cov
from .. import results as res
from .. import gas as gas

from .kalman import *
from .llm import *

class NLLEV(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** NON-GAUSSIAN LOCAL LEVEL MODEL ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    integ : int (default : 0)
        Specifies how many times to difference the time series.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.
    """

    def __init__(self,data,integ=0,target=None):

        # Initialize TSM object
        super(NLLEV,self).__init__('NLLEV')

        # Parameters
        self.integ = integ
        self.max_lag = 0
        self._param_hide = 0 # Whether to cutoff variance parameters from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        self.state_no = 1

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data_original = self.data

        # Difference data
        X = self.data
        for order in range(self.integ):
            X = np.diff(X)
            self.data_name = "Differenced " + self.data_name
        self.data = X       
        self.data_length = X
        self.cutoff = 0

        self._create_parameters()

    def _animate_bbvi(self,stored_parameters,stored_predictive_likelihood):
        """ Produces animated plot of BBVI optimization

        Returns
        ----------
        None (changes model attributes)
        """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ud = BBVINLLMAnimate(ax,self.data,stored_parameters,self.index,self.param_no,self.link)
        anim = FuncAnimation(fig, ud, frames=np.arange(stored_parameters.shape[0]), init_func=ud.init,
                interval=10, blit=True)
        plt.plot(self.data)
        plt.xlabel("Time")
        plt.ylabel(self.data_name)
        plt.show()

    def _create_parameters(self):
        """ Creates model parameters

        Returns
        ----------
        None (changes model attributes)
        """

        self.parameters.add_parameter('Sigma^2 level',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

    def _get_scale_and_shape(self):
        """ Retrieves the scale and shape for the model

        Returns
        ----------
        Scale (float) and shape (float)
        """
        if self.dist == 't':
            return self.parameters.get_parameter_values(transformed=True)[-2],self.parameters.get_parameter_values(transformed=True)[-1],0
        elif self.dist == 'Laplace':
            return self.parameters.get_parameter_values(transformed=True)[-1],0,0
        elif self.dist == 'skewt':
            return self.parameters.get_parameter_values(transformed=True)[-2],self.parameters.get_parameter_values(transformed=True)[-1],self.parameters.get_parameter_values(transformed=True)[-3]
        else:
            return 0, 0, 0

    def _model(self,data,beta):
        """ Creates the structure of the model

        Parameters
        ----------
        data : np.array
            Contains the time series

        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        a,P,K,F,v : np.array
            Filted states, filtered variances, Kalman gains, F matrix, residuals
        """     

        T, Z, R, Q, H = self._ss_matrices(beta)

        return univariate_kalman(data,Z,H,T,Q,R,0.0)

    def _ss_matrices(self,beta):
        """ Creates the state space matrices required

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm
        """     

        T = np.identity(1)
        R = np.identity(1)
        Z = np.identity(1)
        Q = np.identity(1)*self.parameters.parameter_list[0].prior.transform(beta[0])

        return T, Z, R, Q

    def _general_approximating_model(self,beta,T,Z,R,Q,h_approx):
        """ Creates simplest kind of approximating Gaussian model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm

        h_approx : float
            Value to use for the H matrix

        Returns
        ----------

        H : np.array
            Approximating measurement variance matrix

        mu : np.array
            Approximating measurement constants
        """     

        H = np.ones(self.data.shape[0])*h_approx
        mu = np.zeros(self.data.shape[0])

        return H, mu

    def _poisson_approximating_model(self,beta,T,Z,R,Q):
        """ Creates approximating Gaussian model for Poisson measurement density

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm

        Returns
        ----------

        H : np.array
            Approximating measurement variance matrix

        mu : np.array
            Approximating measurement constants
        """     

        if hasattr(self, 'H'):
            H = self.H
        else:
            H = np.ones(self.data.shape[0])

        if hasattr(self, 'mu'):
            mu = self.mu
        else:
            mu = np.zeros(self.data.shape[0])

        alpha = np.array([np.zeros(self.data.shape[0])])
        tol = 100.0
        it = 0
        while tol > 10**-7 and it < 5:
            old_alpha = alpha[0]
            alpha, V = nl_univariate_KFS(self.data,Z,H,T,Q,R,mu)
            H = np.exp(-alpha[0])
            mu = self.data - alpha[0] - np.exp(-alpha[0])*(self.data - np.exp(alpha[0]))
            tol = np.mean(np.abs(alpha[0]-old_alpha))
            it += 1

        return H, mu

    def _skewt_approximating_model(self,beta,T,Z,R,Q):
        """ Creates approximating Gaussian model for skewt measurement density

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm

        Returns
        ----------

        H : np.array
            Approximating measurement variance matrix

        mu : np.array
            Approximating measurement constants
        """     

        H = np.ones(self.data.shape[0])*self.parameters.parameter_list[-2].prior.transform(beta[-2])
        mu = np.zeros(self.data.shape[0])

        return H, mu

    def _t_approximating_model(self,beta,T,Z,R,Q):
        """ Creates approximating Gaussian model for t measurement density

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm

        Returns
        ----------

        H : np.array
            Approximating measurement variance matrix

        mu : np.array
            Approximating measurement constants
        """     

        H = np.ones(self.data.shape[0])*self.parameters.parameter_list[1].prior.transform(beta[1])
        mu = np.zeros(self.data.shape[0])

        return H, mu

    @classmethod
    def Exponential(cls,data,integ=0,target=None):
        """ Creates Exponential-distributed state space model

        Parameters
        ----------
        data : np.array
            Contains the time series

        integ : int (default : 0)
            Specifies how many time to difference the time series.

        target : str (pd.DataFrame) or int (np.array)
            Specifies which column name or array index to use. By default, first
            column/array will be selected as the dependent variable.

        Returns
        ----------
        - NLLEV.Exponential object
        """     

        x = NLLEV(data=data,integ=integ,target=target)
        x.meas_likelihood = x.exponential_likelihood
        x.model_name = "Exponential Local Level Model"  
        x.dist = "Exponential"
        x.param_no = 1  
        x.link = np.exp
        temp = LLEV(data,integ=integ,target=target)
        temp.fit()
        x.parameters.set_parameter_starting_values(np.array([temp.parameters.get_parameter_values()[1]]))

        def approx_model(beta,T,Z,R,Q):
            return x._general_approximating_model(beta,T,Z,R,Q,temp.parameters.get_parameter_values(transformed=True)[0])

        x._approximating_model = approx_model

        def draw_variable(loc,scale,shape,skewness,nsims):
            return np.random.exponential(1/loc, nsims)

        x.draw_variable = draw_variable
        x.m_likelihood_markov_blanket = x.exponential_likelihood_markov_blanket

        return x

    @classmethod
    def Laplace(cls,data,integ=0,target=None):
        """ Creates Laplace-distributed state space model

        Parameters
        ----------
        data : np.array
            Contains the time series

        integ : int (default : 0)
            Specifies how many time to difference the time series.

        target : str (pd.DataFrame) or int (np.array)
            Specifies which column name or array index to use. By default, first
            column/array will be selected as the dependent variable.

        Returns
        ----------
        - NLLEV.Laplace object
        """     

        x = NLLEV(data=data,integ=integ,target=target)
        
        x.parameters.add_parameter('Laplace Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        x.meas_likelihood = x.laplace_likelihood
        x.model_name = "Laplace Local Level Model"
        x.dist = "Laplace"
        x.param_no = 2  
        x.link = np.array
        temp = LLEV(data,integ=integ,target=target)
        temp.fit()
        x.parameters.set_parameter_starting_values(np.array([temp.parameters.get_parameter_values()[1],2]))
        
        def approx_model(beta,T,Z,R,Q):
            return x._general_approximating_model(beta,T,Z,R,Q,temp.parameters.get_parameter_values(transformed=True)[0])

        x._approximating_model = approx_model

        def draw_variable(loc,scale,shape,skewness,nsims):
            return np.random.laplace(loc, scale, nsims)

        x.draw_variable = draw_variable
        x.m_likelihood_markov_blanket = x.laplace_likelihood_markov_blanket

        return x

    @classmethod
    def Poisson(cls,data,integ=0,target=None):
        """ Creates Poisson-distributed state space model

        Parameters
        ----------
        data : np.array
            Contains the time series

        integ : int (default : 0)
            Specifies how many time to difference the time series.

        target : str (pd.DataFrame) or int (np.array)
            Specifies which column name or array index to use. By default, first
            column/array will be selected as the dependent variable.

        Returns
        ----------
        - NLLEV.Poisson object
        """     

        x = NLLEV(data=data,integ=integ,target=target)
        x._approximating_model = x._poisson_approximating_model
        x.meas_likelihood = x.poisson_likelihood
        x.model_name = "Poisson Local Level Model"  
        x.dist = "Poisson"
        x.param_no = 1  
        x.link = np.exp
        temp = LLEV(data,integ=integ,target=target)
        temp.fit()
        x.parameters.set_parameter_starting_values(np.array([temp.parameters.get_parameter_values()[1]]))

        def draw_variable(loc,scale,shape,skewness,nsims):
            return np.random.poisson(loc, nsims)

        x.draw_variable = draw_variable
        x.m_likelihood_markov_blanket = x.poisson_likelihood_markov_blanket

        return x

    @classmethod
    def skewt(cls,data,integ=0,target=None):
        """ Creates skewt-distributed state space model

        Parameters
        ----------
        data : np.array
            Contains the time series

        integ : int (default : 0)
            Specifies how many time to difference the time series.

        target : str (pd.DataFrame) or int (np.array)
            Specifies which column name or array index to use. By default, first
            column/array will be selected as the dependent variable.

        Returns
        ----------
        - NLLEV.skewt object
        """     

        x = NLLEV(data=data,integ=integ,target=target)
        
        x.parameters.add_parameter('Skewness',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        x.parameters.add_parameter('Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        x.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        x._approximating_model = x._skewt_approximating_model
        x.meas_likelihood = x.skewt_likelihood
        x.model_name = "skewt-distributed Local Level Model"
        x.dist = "skewt"
        x.param_no = 4  
        x.link = np.array
        temp = LLEV(data,integ=integ,target=target)
        temp.fit()

        def temp_function(params):
            return -np.sum(dst.skewt.logpdf(x=x.data,df=np.exp(params[0]),
                loc=np.ones(x.data.shape[0])*params[1], scale=np.exp(params[2]),gamma=np.exp(params[3])))

        p = optimize.minimize(temp_function,np.array([2.0,0.0,-1.0,0.0]),method='L-BFGS-B')

        x.parameters.set_parameter_starting_values(np.array([temp.parameters.get_parameter_values()[1],p.x[3],p.x[2],p.x[0]]))

        def draw_variable(loc,scale,shape,skewness,nsims):
            return loc + scale*dst.skewt.rvs(shape,skewness,nsims)

        x.draw_variable = draw_variable
        x.m_likelihood_markov_blanket = x.skewt_likelihood_markov_blanket

        return x

    @classmethod
    def t(cls,data,integ=0,target=None):
        """ Creates t-distributed state space model

        Parameters
        ----------
        data : np.array
            Contains the time series

        integ : int (default : 0)
            Specifies how many time to difference the time series.

        target : str (pd.DataFrame) or int (np.array)
            Specifies which column name or array index to use. By default, first
            column/array will be selected as the dependent variable.

        Returns
        ----------
        - NLLEV.t object
        """     

        x = NLLEV(data=data,integ=integ,target=target)
        
        x.parameters.add_parameter('Scale',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))
        x.parameters.add_parameter('v',ifr.Uniform(transform='exp'),dst.q_Normal(0,3))

        x._approximating_model = x._t_approximating_model
        x.meas_likelihood = x.t_likelihood
        x.model_name = "t-distributed Local Level Model"
        x.dist = "t"
        x.param_no = 3  
        x.link = np.array
        temp = LLEV(data,integ=integ,target=target)
        temp.fit()

        def temp_function(params):
            return -np.sum(ss.t.logpdf(x=x.data,df=np.exp(params[0]),
                loc=np.ones(x.data.shape[0])*params[1], scale=np.exp(params[2])))

        p = optimize.minimize(temp_function,np.array([2.0,0.0,-1.0]),method='L-BFGS-B')

        x.parameters.set_parameter_starting_values(np.array([temp.parameters.get_parameter_values()[1],p.x[2],p.x[0]]))

        def draw_variable(loc,scale,shape,skewness,nsims):
            return loc + scale*np.random.standard_t(shape,nsims)

        x.draw_variable = draw_variable
        x.m_likelihood_markov_blanket = x.t_likelihood_markov_blanket

        return x

    def neg_logposterior(self,beta):
        """ Returns negative log posterior

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            State matrix

        Returns
        ----------
        Negative log posterior
        """
        post = self.neg_loglik(beta)
        for k in range(0,self.param_no):
            post += -self.parameters.parameter_list[k].prior.logpdf(beta[k])
        return post     

    def state_likelihood_markov_blanket(self,beta,alpha,col_no):
        """ Returns Markov blanket of the states given the evolution parameters

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            State matrix

        Returns
        ----------
        State likelihood
        """     
        _, _, _, Q = self._ss_matrices(beta)
        state_terms = np.append(0,ss.norm.logpdf(alpha[col_no][1:]-alpha[col_no][:-1],loc=0,scale=np.power(Q[col_no][col_no],0.5)))
        blanket = state_terms
        blanket[:-1] = blanket[:-1] + blanket[1:]
        return blanket

    def state_likelihood(self,beta,alpha):
        """ Returns likelihood of the states given the evolution parameters

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            State matrix

        Returns
        ----------
        State likelihood
        """

        _, _, _, Q = self._ss_matrices(beta)
        residuals = alpha[0][1:]-alpha[0][:-1]
        return np.sum(ss.norm.logpdf(residuals,loc=0,scale=np.power(Q.ravel(),0.5)))

    def loglik(self,beta,alpha):
        """ Creates loglikelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Loglikelihood
        """     

        return (self.state_likelihood(beta,alpha) + self.meas_likelihood(beta,alpha))

    def neg_loglik(self,beta):
        """ Creates negative loglikelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        Negative loglikelihood
        """     

        states = np.zeros([self.state_no, self.data.shape[0]])
        states[0,:] = beta[self.param_no:self.param_no+self.data.shape[0]] 
        return -self.loglik(beta[:self.param_no],states) 

    def fit(self,optimizer='RMSProp',iterations=1000,print_progress=True,start_diffuse=False,**kwargs):
        """ Fits the model

        Parameters
        ----------
        optimizer : string
            Stochastic optimizer: either RMSProp or ADAM.

        iterations: int
            How many iterations to run

        print_progress : bool
            Whether tp print the ELBO progress or not
        
        start_diffuse : bool
            Whether to start from diffuse values (if not: use approx Gaussian)
        
        Returns
        ----------
        BBVI fit object
        """     

        return self._bbvi_fit(self.neg_logposterior,optimizer=optimizer,print_progress=print_progress,
            start_diffuse=start_diffuse,iterations=iterations,**kwargs)

    def _bbvi_fit(self,posterior,optimizer='RMSProp',iterations=1000,print_progress=True,start_diffuse=False,**kwargs):
        """ Performs Black Box Variational Inference

        Parameters
        ----------
        posterior : method
            Hands bbvi_fit a posterior object

        optimizer : string
            Stochastic optimizer: either RMSProp or ADAM.

        iterations: int
            How many iterations to run

        print_progress : bool
            Whether tp print the ELBO progress or not
        
        start_diffuse : bool
            Whether to start from diffuse values (if not: use approx Gaussian)

        Returns
        ----------
        BBVIResults object
        """

        animate = kwargs.get('animate',False)

        # Starting parameters
        phi = self.parameters.get_parameter_starting_values()

        # Starting values for approximate distribution
        for i in range(len(self.parameters.parameter_list)):
            approx_dist = self.parameters.parameter_list[i].q
            if isinstance(approx_dist, dst.q_Normal):
                self.parameters.parameter_list[i].q.loc = phi[i]
                self.parameters.parameter_list[i].q.scale = -3.0

        q_list = [k.q for k in self.parameters.parameter_list]

        # Get starting values for states
        T, Z, R, Q = self._ss_matrices(phi)
        H, mu = self._approximating_model(phi,T,Z,R,Q)
        a, V = self.smoothed_state(self.data,phi,H,mu)
        V[0][0][0] = V[0][0][-1] 

        for item in range(self.data.shape[0]):
            if start_diffuse is False:
                q_list.append(dst.q_Normal(a[0][item],np.log(np.sqrt(np.abs(V[0][0][item])))))
            else:
                q_list.append(dst.q_Normal(0,np.log(np.sqrt(np.abs(V[0][0][item])))))

        # PERFORM BBVI
        bbvi_obj = ifr.CBBVI(posterior,self.log_p_blanket,q_list,24,optimizer,iterations)

        if print_progress is False:
            bbvi_obj.printer = False

        if animate is True:
            q, q_params, q_ses, stored_parameters, stored_predictive_likelihood = bbvi_obj.run_and_store()
            self._animate_bbvi(stored_parameters,stored_predictive_likelihood)
        else:
            q, q_params, q_ses = bbvi_obj.run()

        self.parameters.set_parameter_values(q_params[:self.param_no],'BBVI',np.exp(q_ses[:self.param_no]),None)    

        # STORE RESULTS
        for k in range(len(self.parameters.parameter_list)):
            self.parameters.parameter_list[k].q = q[k]

        theta = q_params[self.param_no:]
        Y = self.data
        scores = None
        states = q_params[self.param_no:]
        X_names = None
        states_ses = np.exp(q_ses[self.param_no:])

        self.states = states
        self.states_ses = states_ses

        return res.BBVISSResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, parameters=self.parameters,data=Y,index=self.index,
            multivariate_model=self.multivariate_model,objective=posterior(q_params), 
            method='BBVI',ses=q_ses[:self.param_no],signal=theta,scores=scores,
            param_hide=self._param_hide,max_lag=self.max_lag,states=states,states_var=np.power(states_ses,2))

    def exponential_likelihood(self,beta,alpha):
        """ Creates Exponential loglikelihood of the data given the states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Exponential loglikelihood
        """     
        return np.sum(ss.expon.logpdf(self.data,1/np.exp(alpha[0])))

    def exponential_likelihood_markov_blanket(self,beta,alpha):
        """ Creates Expnonential Markov blanket for each state

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Exponential loglikelihood
        """     
        return ss.expon.logpdf(self.data,1/np.exp(alpha[0]))

    def laplace_likelihood(self,beta,alpha):
        """ Creates Poisson loglikelihood of the data given the states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Laplace loglikelihood
        """     
        return np.sum(ss.laplace.logpdf(self.data,alpha[0],scale=self.parameters.parameter_list[-1].prior.transform(beta[-1])))

    def laplace_likelihood_markov_blanket(self,beta,alpha):
        """ Creates Laplace Markov blanket for each state

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Laplace loglikelihood
        """     
        return ss.laplace.logpdf(self.data,alpha[0],scale=self.parameters.parameter_list[-1].prior.transform(beta[-1]))

    def poisson_likelihood(self,beta,alpha):
        """ Creates Poisson loglikelihood of the data given the states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Poisson loglikelihood
        """     
        return np.sum(ss.poisson.logpmf(self.data,np.exp(alpha[0])))

    def poisson_likelihood_markov_blanket(self,beta,alpha):
        """ Creates Poisson Markov blanket for each state

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Poisson loglikelihood
        """     
        return ss.poisson.logpmf(self.data,np.exp(alpha[0]))

    def t_likelihood(self,beta,alpha):
        """ Creates t loglikelihood of the date given the states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        t loglikelihood
        """     
        return np.sum(ss.t.logpdf(x=self.data,
            df=self.parameters.parameter_list[2].prior.transform(beta[2]),
            loc=alpha[0],
            scale=self.parameters.parameter_list[1].prior.transform(beta[1])))

    def t_likelihood_markov_blanket(self,beta,alpha):
        """ Creates t Markov blanket for each state

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Poisson loglikelihood
        """     
        return ss.t.logpdf(x=self.data,
            df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=alpha[0],
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2]))

    def skewt_likelihood(self,beta,alpha):
        """ Creates t loglikelihood of the date given the states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        t loglikelihood
        """     
        return np.sum(dst.skewt.logpdf(x=self.data,
            df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=alpha[0],
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2]),gamma=self.parameters.parameter_list[-3].prior.transform(beta[-3])))

    def skewt_likelihood_markov_blanket(self,beta,alpha):
        """ Creates t Markov blanket for each state

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Poisson loglikelihood
        """     
        return dst.skewt.logpdf(x=self.data,
            df=self.parameters.parameter_list[-1].prior.transform(beta[-1]),
            loc=alpha[0],
            scale=self.parameters.parameter_list[-2].prior.transform(beta[-2]),
            gamma=self.parameters.parameter_list[-3].prior.transform(beta[-3]))

    def markov_blanket(self,beta,alpha):
        """ Creates total Markov blanket for states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Markov blanket for states
        """             
        likelihood_blanket = self.m_likelihood_markov_blanket(beta,alpha)
        state_blanket = self.state_likelihood_markov_blanket(beta,alpha,0)
        for i in range(self.state_no-1):
            likelihood_blanket = np.append(likelihood_blanket,self.m_likelihood_markov_blanket(beta,alpha))
            state_blanket = np.append(state_blanket,self.state_likelihood_markov_blanket(beta,alpha,i+1))
        return likelihood_blanket + state_blanket
        
    def evo_blanket(self,beta,alpha):
        """ Creates Markov blanket for the evolution parameters

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        alpha : np.array
            A vector of states

        Returns
        ----------
        Markov blanket for evolution parameters
        """                 
        evo_blanket = np.zeros(self.state_no)
        for i in range(evo_blanket.shape[0]):
            evo_blanket[i] = self.state_likelihood_markov_blanket(beta,alpha,i).sum()

        if self.dist in ['t']:
            evo_blanket = np.append([self.m_likelihood_markov_blanket(beta,alpha).sum()]*2,evo_blanket)
        elif self.dist in ['skewt']:
            evo_blanket = np.append([self.m_likelihood_markov_blanket(beta,alpha).sum()]*3,evo_blanket)
        elif self.dist in ['Laplace']:
            evo_blanket = np.append([self.m_likelihood_markov_blanket(beta,alpha).sum()],evo_blanket)

        return evo_blanket

    def log_p_blanket(self,beta):
        """ Creates complete Markov blanket for parameters

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        Markov blanket for parameters
        """             
        states = np.zeros([self.state_no, self.data.shape[0]])
        for state_i in range(self.state_no):
            states[state_i,:] = beta[(self.param_no + (self.data.shape[0]*state_i)):(self.param_no + (self.data.shape[0]*(state_i+1)))]     
        
        return np.append(self.evo_blanket(beta,states),self.markov_blanket(beta,states))

    def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):      
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show 95% prediction intervals for the forecast?

        Returns
        ----------
        - Plot of the forecast
        """     

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:
            # Retrieve data, dates and (transformed) parameters
            scale, shape, skewness = self._get_scale_and_shape()
            previous_value = self.data[-1]  
            forecasted_values = np.ones(h)*self.states[-1]  
            date_index = self.shift_dates(h)
            simulations = 10000
            sim_vector = np.zeros([simulations,h])
            t_params = self.transform_parameters()

            for n in range(0,simulations):  
                rnd_q = np.random.normal(0,np.sqrt(self.parameters.get_parameter_values(transformed=True)[0]),h) 
                exp = forecasted_values.copy()

                for t in range(0,h):
                    if t == 0:
                        exp[t] = forecasted_values[t] + rnd_q[t]
                    else:
                        exp[t] = exp[t-1] + rnd_q[t]

                sim_vector[n] = self.draw_variable(loc=self.link(exp),shape=shape,scale=scale,skewness=skewness,nsims=exp.shape[0])

            sim_vector = np.transpose(sim_vector)

            forecasted_values = self.link(forecasted_values)

            if self.dist == 'skewt':
                forecasted_values = forecasted_values + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*gas.SkewtScore.tv_variate_exp(t_params[-1]))

            plt.figure(figsize=figsize) 

            if intervals == True:
                plt.fill_between(date_index[-h-1:], np.insert([np.percentile(i,5) for i in sim_vector],0,previous_value), 
                    np.insert([np.percentile(i,95) for i in sim_vector],0,previous_value), alpha=0.2,label="95 C.I.")   

            plot_values = np.append(self.data[-past_values:],forecasted_values)
            plot_index = date_index[-h-past_values:]

            plt.plot(plot_index,plot_values,label=self.data_name)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
            plt.show()

    def plot_fit(self,intervals=True,**kwargs):
        """ Plots the fit of the model

        Returns
        ----------
        None (plots data and the fit)
        """

        figsize = kwargs.get('figsize',(10,7))

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:
            date_index = copy.deepcopy(self.index)
            date_index = date_index[self.integ:self.data_original.shape[0]+1]
            t_params = self.transform_parameters()

            if self.dist == 'skewt':
                states_upper_95 = (self.states + 1.98*np.sqrt(self.states_ses)) + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*gas.SkewtScore.tv_variate_exp(t_params[-1]))
                states_lower_95 = (self.states - 1.98*np.sqrt(self.states_ses)) + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*gas.SkewtScore.tv_variate_exp(t_params[-1])) 
                mean_states = self.states + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*gas.SkewtScore.tv_variate_exp(t_params[-1]))
            else:
                states_upper_95 = (self.states + 1.98*np.sqrt(self.states_ses)) 
                states_lower_95 = (self.states - 1.98*np.sqrt(self.states_ses))
                mean_states = self.states

            plt.figure(figsize=figsize) 
            
            plt.subplot(2, 1, 1)
            plt.title(self.data_name + " Raw and Smoothed") 

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, self.link(states_lower_95), self.link(states_upper_95), alpha=0.15,label='95% C.I.')   

            plt.plot(date_index,self.data,label='Data')
            plt.plot(date_index,self.link(mean_states),label='Smoothed',c='black')          
            plt.legend(loc=2)
            
            plt.subplot(2, 1, 2)
            plt.title(self.data_name + " Local Level")  

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, self.link(states_lower_95), self.link(states_upper_95), alpha=0.15,label='95% C.I.')   

            plt.plot(date_index,self.link(mean_states),label='Smoothed State')
            plt.legend(loc=2)
            plt.show()

    def predict(self,h=5):      
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predictions
        """     

        if self.parameters.estimated is False:
            raise Exception("No parameters estimated!")
        else:
            # Retrieve data, dates and (transformed) parameters         
            date_index = self.shift_dates(h)
            forecasted_values = np.ones(h)*self.states[-1]
            t_params = self.transform_parameters()

            if self.dist == 'skewt':
                forecasted_values = forecasted_values + ((t_params[-3] - (1.0/t_params[-3]))*t_params[-2]*gas.SkewtScore.tv_variate_exp(t_params[-1]))

            result = pd.DataFrame(self.link(forecasted_values))
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result

    def predict_is(self,h=5):
        """ Makes dynamic in-sample predictions with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        predictions = []

        for t in range(0,h):
            if self.dist == 'Poisson':
                x = NLLEV.Poisson(integ=self.integ,data=self.data_original[:(-h+t)])
            elif self.dist == 't':
                x = NLLEV.t(integ=self.integ,data=self.data_original[:(-h+t)])
            elif self.dist == 'Laplace':
                x = NLLEV.Laplace(integ=self.integ,data=self.data_original[:(-h+t)])
            elif self.dist == 'Exponential':
                x = NLLEV.Exponential(integ=self.integ,data=self.data_original[:(-h+t)])                                
            x.fit(print_progress=False)
            if t == 0:
                predictions = x.predict(1)
            else:
                predictions = pd.concat([predictions,x.predict(1)])
        
        predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

    def plot_predict_is(self,h=5,**kwargs):
        """ Plots forecasts with the estimated model against data
            (Simulated prediction with data)

        Parameters
        ----------
        h : int (default : 5)
            How many steps to forecast

        Returns
        ----------
        - Plot of the forecast against data 
        """     

        figsize = kwargs.get('figsize',(10,7))

        plt.figure(figsize=figsize)
        date_index = self.index[-h:]
        predictions = self.predict_is(h)
        data = self.data[-h:]

        plt.plot(date_index,data,label='Data')
        plt.plot(date_index,predictions,label='Predictions',c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def simulation_smoother(self,beta):
        """ Durbin and Koopman simulation smoother - simulates from states 
        given model parameters and observations

        Parameters
        ----------

        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        - A simulated state evolution
        """         

        T, Z, R, Q = self._ss_matrices(beta)
        H, mu = self._approximating_model(beta,T,Z,R,Q)

        # Generate e_t+ and n_t+
        rnd_h = np.random.normal(0,np.sqrt(H),self.data.shape[0])
        q_dist = ss.multivariate_normal([0.0], Q)
        rnd_q = q_dist.rvs(self.data.shape[0])

        # Generate a_t+ and y_t+
        a_plus = np.zeros((T.shape[0],self.data.shape[0])) 
        y_plus = np.zeros(self.data.shape[0])

        for t in range(0,self.data.shape[0]):
            if t == 0:
                a_plus[:,t] = np.dot(T,a_plus[:,t]) + rnd_q[t]
                y_plus[t] = mu[t] + np.dot(Z,a_plus[:,t]) + rnd_h[t]
            else:
                if t != self.data.shape[0]:
                    a_plus[:,t] = np.dot(T,a_plus[:,t-1]) + rnd_q[t]
                    y_plus[t] = mu[t] + np.dot(Z,a_plus[:,t]) + rnd_h[t]

        alpha_hat, _ = self.smoothed_state(self.data,beta, H, mu)
        alpha_hat_plus, _ = self.smoothed_state(y_plus,beta, H, mu)
        alpha_tilde = alpha_hat - alpha_hat_plus + a_plus
        
        return alpha_tilde

    def smoothed_state(self,data,beta, H, mu):
        """ Creates smoothed state estimate given state matrices and 
        parameters.

        Parameters
        ----------

        data : np.array
            Data to be smoothed

        beta : np.array
            Contains untransformed starting values for parameters

        Returns
        ----------
        - Smoothed states
        """         

        T, Z, R, Q = self._ss_matrices(beta)
        alpha, V = nl_univariate_KFS(data,Z,H,T,Q,R,mu)
        return alpha, V


# TO DO - INTEGRATE THIS INTO EXISTING CODE MORE CLEANLY

class BBVINLLMAnimate(object):
    def __init__(self,ax,data,means,index,start_index,link):
        self.data = data
        self.line, = ax.plot([], [], 'k-')
        self.index = index
        self.ax = ax
        self.ax.set_xlim(0, data.shape[0])
        self.ax.set_ylim(np.min(data)-0.1*np.std(data), np.max(data)+0.1*np.std(data))
        self.start_index = start_index
        self.means = means
        self.link = link

    def init(self):
        self.line.set_data(range(int(self.means[0].shape[0]-self.start_index)),self.link(self.means[0][self.start_index:]))
        return self.line,

    def __call__(self, i):
        if i == 0:
            return self.init()
        else:
            self.line.set_data(range(int(self.means[0].shape[0]-self.start_index)),self.link(self.means[i][self.start_index:]))
        return self.line,