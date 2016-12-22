import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import optimize

from .. import inference as ifr
from .. import families as fam
from .. import output as op
from .. import tsm as tsm
from .. import data_check as dc
from .. import covariances as cov
from .. import results as res
from .. import gas as gas

from .kalman import *
from .llt import *

class NLLT(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** NON-GAUSSIAN LOCAL LINEAR TREND MODEL ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    integ : int (default : 0)
        Specifies how many time to difference the time series.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.

    family : 
        e.g. pf.Normal(0,1)
    """

    def __init__(self, data, family, integ=0, target=None):

        # Initialize TSM object
        super(NLLT,self).__init__('NLLT')

        # Latent Variables
        self.integ = integ
        self.target = target
        self.max_lag = 0
        self._z_hide = 0 # Whether to cutoff variance latent variables from results
        self.supported_methods = ["MLE", "PML", "Laplace", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        self.state_no = 2

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data = self.data.astype(np.float)
        self.data_original = self.data

        # Difference data
        X = self.data
        for order in range(self.integ):
            X = np.diff(X)
            self.data_name = "Differenced " + self.data_name
        self.data = X       
        self.cutoff = 0
        self.data_length = self.data.shape[0]

        self._create_latent_variables()
        self.family = family
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, self.cythonized = self.family.setup()

        self.model_name = self.model_name2 + " Local Linear Trend Model"

        # Build any remaining latent variables that are specific to the family chosen
        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0],i[1],i[2])
            self.latent_variables.z_list[no+1].start = i[3]

        self.z_no = len(self.latent_variables.z_list)


    def _get_scale_and_shape(self, parm):
        """ Obtains appropriate model scale and shape latent variables

        Parameters
        ----------
        parm : np.array
            Transformed latent variables vector

        Returns
        ----------
        None (changes model attributes)
        """

        if self.scale is True:
            if self.shape is True:
                model_shape = parm[-1]  
                model_scale = parm[-2]
            else:
                model_shape = 0
                model_scale = parm[-1]
        else:
            model_scale = 0
            model_shape = 0 

        if self.skewness is True:
            model_skewness = parm[-3]
        else:
            model_skewness = 0

        return model_scale, model_shape, model_skewness

    def neg_loglik(self, beta):
        """ Creates negative loglikelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - Negative loglikelihood
        """     
        Z = np.zeros(2)
        Z[0] = 1          
        states = np.zeros([self.state_no, self.data.shape[0]])
        states[0,:] = beta[self.z_no:self.z_no+self.data.shape[0]] 
        states[1,:] = beta[self.z_no+self.data.shape[0]:] 
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(self.z_no)]) # transformed distribution parameters
        scale, shape, skewness = self._get_scale_and_shape(parm)
        return self.state_likelihood(beta, states) + self.family.neg_loglikelihood(self.data, self.link(np.dot(Z, states)), scale, shape, skewness)  # negative loglikelihood for model

    def likelihood_markov_blanket(self, beta):
        """ Creates likelihood markov blanket of the modeel

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - Negative loglikelihood
        """     
        states = np.zeros([self.state_no, self.data_length])
        for state_i in range(self.state_no):
            states[state_i,:] = beta[(self.z_no + (self.data_length*state_i)):(self.z_no + (self.data_length*(state_i+1)))]     
        
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(self.z_no)]) # transformed distribution parameters
        scale, shape, skewness = self._get_scale_and_shape(parm)
        Z = np.zeros(2)
        Z[0] = 1     

        return self.family.markov_blanket(self.data, self.link(np.dot(Z, states)), scale, shape, skewness)  # negative loglikelihood for model

    def state_likelihood(self, beta, alpha):
        """ Returns likelihood of the states given the variance latent variables

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        alpha : np.array
            State matrix
        
        Returns
        ----------
        State likelihood
        """
        _, _, _, Q = self._ss_matrices(beta)
        residuals_1 = alpha[0][1:alpha[0].shape[0]]-alpha[0][0:alpha[0].shape[0]-1]
        residuals_2 = alpha[1][1:alpha[1].shape[0]]-alpha[1][0:alpha[1].shape[0]-1]
        return np.sum(ss.norm.logpdf(residuals_1,loc=0,scale=np.power(Q[0][0],0.5))) + np.sum(ss.norm.logpdf(residuals_2,loc=0,scale=np.power(Q[1][1],0.5)))

    def state_likelihood_markov_blanket(self, beta, alpha, col_no):
        """ Returns Markov blanket of the states given the variance latent variables

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        alpha : np.array
            State matrix

        Returns
        ----------
        State likelihood
        """     
        _, _, _, Q = self._ss_matrices(beta)
        blanket = np.append(0,ss.norm.logpdf(alpha[col_no][1:]-alpha[col_no][:-1],loc=0,scale=np.sqrt(Q[col_no][col_no])))
        blanket[:-1] = blanket[:-1] + blanket[1:]
        return blanket

    def neg_logposterior(self, beta):
        """ Returns negative log posterior

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        alpha : np.array
            State matrix

        Returns
        ----------
        - Negative log posterior
        """
        post = self.neg_loglik(beta)
        for k in range(0,self.z_no):
            post += -self.latent_variables.z_list[k].prior.logpdf(beta[k])
        return post     

    def markov_blanket(self, beta, alpha):
        """ Creates total Markov blanket for states

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        alpha : np.array
            A vector of states

        Returns
        ----------
        Markov blanket for states
        """             
        likelihood_blanket = self.likelihood_markov_blanket(beta)
        state_blanket = self.state_likelihood_markov_blanket(beta,alpha,0)
        for i in range(self.state_no-1):
            likelihood_blanket = np.append(likelihood_blanket,self.likelihood_markov_blanket(beta))
            state_blanket = np.append(state_blanket,self.state_likelihood_markov_blanket(beta,alpha,i+1))
        return likelihood_blanket + state_blanket
        
    def evo_blanket(self, beta, alpha):
        """ Creates Markov blanket for the variance latent variables

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        alpha : np.array
            A vector of states

        Returns
        ----------
        Markov blanket for variance latent variables
        """                 

        # Markov blanket for each state
        evo_blanket = np.zeros(self.state_no)
        for i in range(evo_blanket.shape[0]):
            evo_blanket[i] = self.state_likelihood_markov_blanket(beta, alpha, i).sum()

        # If the family has additional parameters, add their markov blankets
        if self.z_no > 2:
            evo_blanket = np.append([self.likelihood_markov_blanket(beta).sum()]*(self.z_no-1),evo_blanket)

        return evo_blanket

    def log_p_blanket(self, beta):
        """ Creates complete Markov blanket for latent variables

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        Markov blanket for latent variables
        """             
        states = np.zeros([self.state_no, self.data_length])
        for state_i in range(self.state_no):
            states[state_i,:] = beta[(self.z_no + (self.data_length*state_i)):(self.z_no + (self.data_length*(state_i+1)))]     
        
        return np.append(self.evo_blanket(beta,states),self.markov_blanket(beta,states))

    def _animate_bbvi(self, stored_parameters, stored_predictive_likelihood):
        """ Produces animated plot of BBVI optimization

        Returns
        ----------
        None (changes model attributes)
        """
        from matplotlib.animation import FuncAnimation, writers
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ud = BBVINLLTAnimate(ax,self.data,stored_parameters,self.index,self.z_no,self.link)
        anim = FuncAnimation(fig, ud, frames=np.arange(stored_parameters.shape[0]), init_func=ud.init,
                interval=10, blit=True)
        plt.plot(self.data)
        plt.xlabel("Time")
        plt.ylabel(self.data_name)
        plt.show()

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        self.latent_variables.add_z('Sigma^2 level', fam.Flat(transform='exp'), fam.Normal(0,3))
        self.latent_variables.add_z('Sigma^2 trend', fam.Flat(transform='exp'), fam.Normal(0,3))

    def _model(self, data, beta):
        """ Creates the structure of the model

        Parameters
        ----------
        data : np.array
            Contains the time series

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        a,P,K,F,v : np.array
            Filted states, filtered variances, Kalman gains, F matrix, residuals
        """     

        T, Z, R, Q, H = self._ss_matrices(beta)

        return univariate_kalman(data,Z,H,T,Q,R,0.0)

    def _preoptimize_model(self):
        """ Preoptimizes the model by estimating a Gaussian state space models
        
        Returns
        ----------
        - Gaussian model latent variable object
        """
        gaussian_model = LLT(self.data, integ=self.integ, target=self.target)
        gaussian_model.fit()
        self.latent_variables.z_list[0].start = gaussian_model.latent_variables.get_z_values()[1]
        self.latent_variables.z_list[1].start = gaussian_model.latent_variables.get_z_values()[2]

        if self.model_name2 == 't':

            def temp_function(params):
                return -np.sum(ss.t.logpdf(x=self.data, df=np.exp(params[0]), 
                    loc=np.ones(self.data.shape[0])*params[1], scale=np.exp(params[2])))

            p = optimize.minimize(temp_function,np.array([2.0, 0.0, -1.0]),method='L-BFGS-B')
            self.latent_variables.z_list[2].start = p.x[2]
            self.latent_variables.z_list[3].start = p.x[0]

        elif self.model_name2 == 'Skewt':

            def temp_function(params):
                return -np.sum(fam.Skewt.logpdf_internal(x=self.data,df=np.exp(params[0]),
                    loc=np.ones(self.data.shape[0])*params[1], scale=np.exp(params[2]),gamma=np.exp(params[3])))

            p = optimize.minimize(temp_function,np.array([2.0, 0.0, -1.0, 0.0]),method='L-BFGS-B')
            self.latent_variables.z_list[2].start = p.x[3]
            self.latent_variables.z_list[3].start = p.x[2]
            self.latent_variables.z_list[4].start = p.x[0]

        return gaussian_model.latent_variables

    def _ss_matrices(self, beta):
        """ Creates the state space matrices required

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        T, Z, R, Q : np.array
            State space matrices used in KFS algorithm
        """     

        T = np.identity(2)
        T[0][1] = 1
        
        Z = np.zeros(2)
        Z[0] = 1

        R = np.identity(2)
        Q = np.identity(2)
        Q[0][0] = self.latent_variables.z_list[0].prior.transform(beta[0])
        Q[1][1] = self.latent_variables.z_list[1].prior.transform(beta[1])

        return T, Z, R, Q

    def _general_approximating_model(self, beta, T, Z, R, Q, h_approx):
        """ Creates simplest kind of approximating Gaussian model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

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

        H = np.ones(self.data_length)*h_approx
        mu = np.zeros(self.data_length)

        return H, mu

    def fit(self, optimizer='RMSProp', iterations=1000, print_progress=True, start_diffuse=False, **kwargs):
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

        return self._bbvi_fit(optimizer=optimizer, print_progress=print_progress,
            start_diffuse=start_diffuse, iterations=iterations, **kwargs)


    def initialize_approx_dist(self, phi, start_diffuse, gaussian_latents):
        """ Initializes the appoximate distibution for the model

        Parameters
        ----------
        phi : np.ndarray
            Latent variables

        start_diffuse: boolean
            Whether to start from diffuse values or not
        
        gaussian_latents: LatentVariables object
            Latent variables for the Gaussian approximation

        Returns
        ----------
        BBVI fit object
        """     

        # Starting values for approximate distribution
        for i in range(len(self.latent_variables.z_list)):
            approx_dist = self.latent_variables.z_list[i].q
            if isinstance(approx_dist, fam.Normal):
                self.latent_variables.z_list[i].q.mu0 = phi[i]
                self.latent_variables.z_list[i].q.sigma0 = np.exp(-3.0)

        q_list = [k.q for k in self.latent_variables.z_list]

        # Get starting values for states
        T, Z, R, Q = self._ss_matrices(phi)
        H, mu = self.family.approximating_model(phi, T, Z, R, Q, gaussian_latents.get_z_values(transformed=True)[0], self.data)
        a, V = self.smoothed_state(self.data, phi, H, mu)

        for item in range(self.data_length):
            if start_diffuse is False:
                q_list.append(fam.Normal(a[0][item], np.std(self.data)))
            else:
                q_list.append(fam.Normal(self.family.itransform(np.mean(self.data)), np.std(self.data)))

        for item in range(self.data_length):  
            if start_diffuse is False:        
                q_list.append(fam.Normal(a[1][item], np.std(self.data)))
            else:
                q_list.append(fam.Normal(self.family.itransform(np.mean(self.data)), np.std(self.data)))

        return q_list

    def _bbvi_fit(self, optimizer='RMSProp', iterations=1000, print_progress=True,
        start_diffuse=False, **kwargs):
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
        if self.model_name2 in ["t", "Skewt"]:
            default_learning_rate = 0.0001
        else:
            default_learning_rate = 0.001

        animate = kwargs.get('animate', False)
        batch_size = kwargs.get('batch_size', 24) 
        learning_rate = kwargs.get('learning_rate', default_learning_rate) 
        record_elbo = kwargs.get('record_elbo', False) 

        # Starting values
        gaussian_latents = self._preoptimize_model() # find parameters for Gaussian model
        phi = self.latent_variables.get_z_starting_values()
        q_list = self.initialize_approx_dist(phi, start_diffuse, gaussian_latents)

        # PERFORM BBVI
        bbvi_obj = ifr.CBBVI(self.neg_logposterior, self.log_p_blanket, q_list, batch_size, 
            optimizer, iterations, learning_rate, record_elbo)

        if print_progress is False:
            bbvi_obj.printer = False

        if animate is True:
            q, q_params, q_ses, stored_z, stored_predictive_likelihood = bbvi_obj.run_and_store()
            self._animate_bbvi(stored_z,stored_predictive_likelihood)
        else:
            q, q_params, q_ses, elbo_records = bbvi_obj.run()

        self.latent_variables.set_z_values(q_params[:self.z_no],'BBVI',np.exp(q_ses[:self.z_no]),None)    

        # STORE RESULTS
        for k in range(len(self.latent_variables.z_list)):
            self.latent_variables.z_list[k].q = q[k]

        theta = q_params[self.z_no:self.z_no+self.data_length]

        Y = self.data
        scores = None
        states = np.array([q_params[self.z_no:self.z_no+self.data_length],
            q_params[self.z_no+self.data_length:]])
        X_names = None
        states_var = np.array([np.exp(q_ses[self.z_no:self.z_no+self.data_length]),
            np.exp(q_ses[self.z_no+self.data_length:])])

        self.states = states
        self.states_var = states_var

        return res.BBVISSResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, latent_variables=self.latent_variables,data=Y,index=self.index,
            multivariate_model=self.multivariate_model,objective=self.neg_logposterior(q_params), 
            method='BBVI',ses=q_ses[:self.z_no],signal=theta,scores=scores, elbo_records=elbo_records,
            z_hide=self._z_hide,max_lag=self.max_lag,states=states,states_var=states_var)

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
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            # Retrieve data, dates and (transformed) latent variables
            scale, shape, skewness = scale, shape, skewness = self._get_scale_and_shape(self.latent_variables.get_z_values(transformed=True))

            # Get expected values
            forecasted_values = np.zeros(h)

            for value in range(0,h):
                if value == 0:
                    forecasted_values[value] = self.states[0][-1] + self.states[1][-1]
                else:
                    forecasted_values[value] = forecasted_values[value-1] + self.states[1][-1]

            previous_value = self.data[-1]  
            date_index = self.shift_dates(h)
            simulations = 10000
            sim_vector = np.zeros([simulations,h])

            for n in range(0,simulations):  
                rnd_q = np.random.normal(0,np.sqrt(self.latent_variables.get_z_values(transformed=True)[0]),h)
                rnd_q2 = np.random.normal(0,np.sqrt(self.latent_variables.get_z_values(transformed=True)[1]),h)
                exp_0 = np.zeros(h)
                exp_1 = np.zeros(h)

                for value in range(0,h):
                    if value == 0:
                        exp_0[value] = self.states[1][-1] + self.states[0][-1] + rnd_q[value]
                        exp_1[value] = self.states[1][-1] + rnd_q2[value]
                    else:
                        exp_0[value] = exp_0[value-1] + exp_1[value-1] + rnd_q[value]
                        exp_1[value] = exp_1[value-1] + rnd_q2[value]

                sim_vector[n] = self.family.draw_variable(loc=self.link(exp_0),shape=shape,scale=scale,skewness=skewness,nsims=exp_0.shape[0])

            sim_vector = np.transpose(sim_vector)
            forecasted_values = self.link(forecasted_values)

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
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            date_index = copy.deepcopy(self.index)
            date_index = date_index[self.integ:self.data_original.shape[0]+1]

            states_0_upper_95 = self.states[0] + 1.98*np.sqrt(self.states_var[0])
            states_0_lower_95 = self.states[0] - 1.98*np.sqrt(self.states_var[0])
            states_1_upper_95 = self.states[1] + 1.98*np.sqrt(self.states_var[1])
            states_1_lower_95 = self.states[1] - 1.98*np.sqrt(self.states_var[1])

            plt.figure(figsize=figsize) 
            
            plt.subplot(2, 2, 1)
            plt.title(self.data_name + " Raw and Smoothed") 

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, self.link(states_0_lower_95), self.link(states_0_upper_95), alpha=0.15,label='95% C.I.')   

            plt.plot(date_index,self.data,label='Data')
            plt.plot(date_index,self.link(self.states[0]),label="Smoothed",c='black')
            plt.legend(loc=2)
            
            plt.subplot(2, 2, 2)
            plt.title(self.data_name + " Local Level")  

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, self.link(states_0_lower_95), self.link(states_0_upper_95), alpha=0.15,label='95% C.I.')   

            plt.plot(date_index,self.link(self.states[0]),label='Smoothed State')
            plt.legend(loc=2)
            
            plt.subplot(2, 2, 3)
            plt.title(self.data_name + " Trend")    

            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                plt.fill_between(date_index, states_1_lower_95, states_1_upper_95, alpha=0.15,label='95% C.I.') 

            plt.plot(date_index,self.states[1],label='Smoothed State')
            plt.legend(loc=2)
            
            plt.show()

    def predict(self, h=5):      
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predictions
        """     

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            # Retrieve data, dates and (transformed) latent variables         
            date_index = self.shift_dates(h)

            # Get expected values
            forecasted_values = np.zeros(h)

            for value in range(0,h):
                if value == 0:
                    forecasted_values[value] = self.states[0][-1] + self.states[1][-1]
                else:
                    forecasted_values[value] = forecasted_values[value-1] + self.states[1][-1]

            result = pd.DataFrame(self.link(forecasted_values))
            result.rename(columns={0:self.data_name}, inplace=True)
            result.index = date_index[-h:]

            return result

    def predict_is(self, h=5, fit_once=True):
        """ Makes dynamic in-sample predictions with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        fit_once : boolean
            (default: True) Fits only once before the in-sample prediction; if False, fits after every new datapoint
            This method is not functional currently for this model

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        predictions = []

        for t in range(0,h):
            x = NLLT(family=self.family, integ=self.integ, data=self.data_original[:(-h+t)])
                           
            x.fit(print_progress=False)

            if t == 0:
                predictions = x.predict(h=1)
            else:
                predictions = pd.concat([predictions,x.predict(h=1)])

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
        import matplotlib.pyplot as plt
        import seaborn as sns

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

    def smoothed_state(self,data,beta, H, mu):
        """ Creates smoothed state estimate given state matrices and 
        latent variables.

        Parameters
        ----------

        data : np.array
            Data to be smoothed

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - Smoothed states
        """         

        T, Z, R, Q = self._ss_matrices(beta)
        alpha, V = nl_univariate_KFS(data,Z,H,T,Q,R,mu)
        return alpha, V


# TO DO - INTEGRATE THIS INTO EXISTING CODE MORE CLEANLY

class BBVINLLTAnimate(object):
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
        self.line.set_data(range(int((self.means[0].shape[0]-self.start_index)/2)),
            self.link(self.means[0][self.start_index:-((self.means[0].shape[0]-self.start_index)/2)]))
        return self.line,

    def __call__(self, i):
        if i == 0:
            return self.init()
        else:
            self.line.set_data(range(int((self.means[0].shape[0]-self.start_index)/2)),
                self.link(self.means[i][self.start_index:-((self.means[0].shape[0]-self.start_index)/2)]))
        return self.line,