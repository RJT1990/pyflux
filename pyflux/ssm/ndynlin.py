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
from .dynlin import *

class NDynReg(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** NON-GAUSSIAN DYNAMIC REGRESSION MODELS ****

    Parameters
    ----------

    formula : string
        patsy string describing the regression

    data : pd.DataFrame
        Field to specify the data that will be used

    family : 
        e.g. pf.Normal(0,1)
    """

    def __init__(self, formula, data, family):

        # Initialize TSM object
        super(NDynReg,self).__init__('NDynReg')

        # Latent Variables
        self.max_lag = 0
        self._z_hide = 0 # Whether to cutoff variance latent variables from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.z_no = self.X.shape[1]
        self.y_name = self.y.design_info.describe()
        self.data_name = self.y_name
        self.X_names = self.X.design_info.describe().split(" + ")
        self.y = np.array([self.y]).ravel()
        self.data = self.y
        self.data_length = self.data.shape[0]
        self.X = np.array([self.X])[0]
        self.index = data.index
        self.state_no = self.X.shape[1]

        self._create_latent_variables()
        self.family = family
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, self.cythonized = self.family.setup()

        self.model_name = self.model_name2 + " Dynamic Regression Model"

        # Build any remaining latent variables that are specific to the family chosen
        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0],i[1],i[2])
            self.latent_variables.z_list[no+1].start = i[3]

        self.family_z_no = len(self.family.build_latent_variables())
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
        states = np.zeros([self.state_no, self.data_length])
        for state_i in range(self.state_no):
            states[state_i,:] = beta[(self.z_no + (self.data_length*state_i)):(self.z_no + (self.data_length*(state_i+1)))]
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(self.z_no)]) # transformed distribution parameters
        scale, shape, skewness = self._get_scale_and_shape(parm)
        return self.state_likelihood(beta, states) + self.family.neg_loglikelihood(self.data, self.link(np.sum(self.X*states.T,axis=1)), scale, shape, skewness)  # negative loglikelihood for model

    def likelihood_markov_blanket(self, beta):
        """ Creates likelihood markov blanket of the model

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
        return self.family.markov_blanket(self.data, self.link(np.sum(self.X*states.T,axis=1)), scale, shape, skewness)  # negative loglikelihood for model

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
        state_lik = 0
        for i in range(alpha.shape[0]):
            state_lik += np.sum(ss.norm.logpdf(alpha[i][1:]-alpha[i][:-1],loc=0,scale=np.power(Q[i][i],0.5))) 
        return state_lik

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
        state_blanket = self.state_likelihood_markov_blanket(beta, alpha, 0)
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
        if self.family_z_no > 0:
            evo_blanket = np.append([self.likelihood_markov_blanket(beta).sum()]*(self.family_z_no),evo_blanket)

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
        states = np.zeros([self.state_no, self.data.shape[0]])
        for state_i in range(self.state_no):
            states[state_i,:] = beta[(self.z_no + (self.data.shape[0]*state_i)):(self.z_no + (self.data.shape[0]*(state_i+1)))]     
        
        return np.append(self.evo_blanket(beta,states),self.markov_blanket(beta,states))

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """
        for parm in range(self.z_no):
            self.latent_variables.add_z('Sigma^2 ' + self.X_names[parm], fam.Flat(transform='exp'), fam.Normal(0,3))

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
        a, P, K, F, v : np.array
            Filted states, filtered variances, Kalman gains, F matrix, residuals
        """     

        T, Z, R, Q, H = self._ss_matrices(beta)

        return nld_univariate_kalman(data, Z, H, T, Q, R, 0.0)

    def _preoptimize_model(self):
        """ Preoptimizes the model by estimating a Gaussian state space models
        
        Returns
        ----------
        - Gaussian model latent variable object
        """
        gaussian_model = DynReg(formula=self.formula, data=self.data_original)
        gaussian_model.fit()

        for i in range(self.z_no-self.family_z_no):
            self.latent_variables.z_list[i].start = gaussian_model.latent_variables.get_z_values()[i+1]

        if self.model_name2 == 't':

            def temp_function(params):
                return -np.sum(ss.t.logpdf(x=self.data, df=np.exp(params[0]), 
                    loc=np.ones(self.data.shape[0])*params[1], scale=np.exp(params[2])))

            p = optimize.minimize(temp_function,np.array([2.0,0.0,-1.0]),method='L-BFGS-B')
            self.latent_variables.z_list[-2].start = p.x[2]
            self.latent_variables.z_list[-1].start = p.x[0]

        elif self.model_name2 == 'Skewt':

            def temp_function(params):
                return -np.sum(fam.Skewt.logpdf_internal(x=self.data,df=np.exp(params[0]),
                    loc=np.ones(self.data.shape[0])*params[1], scale=np.exp(params[2]),gamma=np.exp(params[3])))

            p = optimize.minimize(temp_function,np.array([2.0,0.0,-1.0,0.0]),method='L-BFGS-B')
            self.latent_variables.z_list[-3].start = p.x[3]
            self.latent_variables.z_list[-2].start = p.x[2]
            self.latent_variables.z_list[-1].start = p.x[0]

        return gaussian_model.latent_variables

    def _ss_matrices(self,beta):
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


        T = np.identity(self.state_no)
        Z = self.X
        R = np.identity(self.state_no)
        
        Q = np.identity(self.state_no)
        for i in range(0,self.state_no):
            Q[i][i] = self.latent_variables.z_list[i].prior.transform(beta[i])

        return T, Z, R, Q

    def _general_approximating_model(self,beta,T,Z,R,Q,h_approx):
        """ Creates simplest approximating model

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

        H = np.ones(self.data.shape[0])*h_approx
        mu = np.zeros(self.data.shape[0])

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
                self.latent_variables.z_list[i].q.loc = phi[i]
                self.latent_variables.z_list[i].q.scale = -3.0

        q_list = [k.q for k in self.latent_variables.z_list]

        # Get starting values for states
        T, Z, R, Q = self._ss_matrices(phi)

        H, mu = self.family.approximating_model_reg(phi, T, Z, R, Q, 
            gaussian_latents.get_z_values(transformed=True)[0], self.data, self.X, self.state_no)

        a, V = self.smoothed_state(self.data, phi, H, mu)

        V[0][0][0] = V[0][0][-1] 

        for state in range(self.state_no):
            for item in range(self.data_length):
                if start_diffuse is False:
                    q_list.append(fam.Normal(a[state][item], np.sqrt(np.abs(V[0][state][item]))))
                else:
                    q_list.append(fam.Normal(self.family.itransform(np.mean(self.data)),np.exp(-3)))

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

        q, q_params, q_ses, elbo_records = bbvi_obj.run()

        self.latent_variables.set_z_values(q_params[:self.z_no],'BBVI',np.exp(q_ses[:self.z_no]),None)    

        # STORE RESULTS
        for k in range(len(self.latent_variables.z_list)):
            self.latent_variables.z_list[k].q = q[k]

        # Theta values and states
        states = q_params[self.z_no:self.z_no+self.data.shape[0]]
        states_var = np.exp(q_ses[self.z_no:self.z_no+self.data.shape[0]])

        for state_i in range(1,self.state_no):
            states = np.vstack((states,q_params[(self.z_no+(self.data.shape[0]*state_i)):(self.z_no+(self.data.shape[0]*(state_i+1)))]))
            states_var = np.vstack((states_var,np.exp(q_ses[(self.z_no+(self.data.shape[0]*state_i)):(self.z_no+(self.data.shape[0]*(state_i+1)))])))

        if self.state_no == 1:
            states = np.array([states])
            states_var = np.array([states_var])

        theta = np.sum(self.X*states.T,axis=1)          
        Y = self.data
        scores = None
        X_names = self.X_names
        self.states = states
        self.states_var = states_var

        return res.BBVISSResults(data_name=self.data_name,X_names=X_names,model_name=self.model_name,
            model_type=self.model_type, latent_variables=self.latent_variables,data=Y,index=self.index,
            multivariate_model=self.multivariate_model,objective=self.neg_logposterior(q_params), 
            method='BBVI',ses=q_ses[:self.z_no],signal=theta,scores=scores,elbo_records=elbo_records,
            z_hide=self._z_hide,max_lag=self.max_lag,states=states,states_var=states_var)

    def plot_predict(self, h=5, past_values=20, intervals=True, oos_data=None, **kwargs):        
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show 95% prediction intervals for the forecast?

        oos_data : pd.DataFrame
            OOS data to use; needs to be same format (columns) as original data

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
            scale, shape, skewness = self._get_scale_and_shape(self.latent_variables.get_z_values(transformed=True))

            # Retrieve data, dates and (transformed) latent variables
            date_index = self.shift_dates(h)
            simulations = 10000
            sim_vector = np.zeros([simulations,h])

            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            full_X = self.X.copy()
            full_X = np.append(full_X,X_oos,axis=0)
            Z = full_X
            a = self.states

            # Retrieve data, dates and (transformed) latent variables         
            smoothed_series = np.zeros(h)
            for t in range(h):
                smoothed_series[t] = self.link(np.dot(Z[self.y.shape[0]+t],a[:,-1]))

            for n in range(0,simulations):  
                rnd_q = np.zeros((self.state_no,h))
                coeff_sim = np.zeros((self.state_no,h))

                # TO DO: vectorize this (easy)
                for state in range(self.state_no):
                    rnd_q[state] = np.random.normal(0,np.sqrt(self.latent_variables.get_z_values(transformed=True)[state]),h)

                for t in range(0,h):
                    if t == 0:
                        for state in range(self.state_no):
                            coeff_sim[state][t] = a[state][-1] + rnd_q[state][t]
                    else:
                        for state in range(self.state_no):
                            coeff_sim[state][t] = coeff_sim[state][t-1] + rnd_q[state][t]

                sim_vector[n] = self.family.draw_variable(loc=self.link(np.sum(coeff_sim.T*Z[self.y.shape[0]:self.y.shape[0]+h,:],axis=1)),shape=shape,scale=scale,skewness=skewness,nsims=h)

            sim_vector = np.transpose(sim_vector)
            forecasted_values = smoothed_series
            previous_value = self.data[-1]

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

    def plot_fit(self, intervals=True, **kwargs):
        """ Plots the fit of the model

        Parameters
        ----------
        intervals : Boolean
            Whether to plot 95% confidence interval of states

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
            date_index = date_index[:self.y.shape[0]+1]

            mu = self.states
            var = self.states_var
            # Create smoothed/filtered aggregate series
            _, Z, _, _ = self._ss_matrices(self.latent_variables.get_z_values())
            smoothed_series = np.zeros(self.y.shape[0])

            for t in range(0,self.y.shape[0]):
                smoothed_series[t] = np.dot(Z[t],mu[:,t])

            plt.figure(figsize=figsize) 
            
            plt.subplot(self.state_no+1, 1, 1)
            plt.title(self.y_name + " Raw and Smoothed")    
            plt.plot(date_index,self.data,label='Data')
            plt.plot(date_index,self.link(smoothed_series),label='Smoothed Series',c='black')
            plt.legend(loc=2)
            
            for coef in range(0,self.state_no):
                V_coef = self.states_var[coef]
                plt.subplot(self.state_no+1, 1, 2+coef)
                plt.title("Beta " + self.X_names[coef]) 
                states_upper_95 = self.states[coef] + 1.98*np.sqrt(V_coef)
                states_lower_95 = self.states[coef] - 1.98*np.sqrt(V_coef)

                if intervals == True:
                    alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                    plt.fill_between(date_index, states_lower_95, states_upper_95, alpha=0.15,label='95% C.I.') 

                plt.plot(date_index,mu[coef,:],label='Coefficient')
                plt.legend(loc=2)               
            
            plt.show()  

    def predict(self,h=5,oos_data=None):        
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        oos_data : pd.DataFrame
            Data for the variables to be used out of sample (ys can be NaNs)

        Returns
        ----------
        - pd.DataFrame with predictions
        """     

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            full_X = self.X.copy()
            full_X = np.append(full_X,X_oos,axis=0)
            Z = full_X
            a = self.states

            # Retrieve data, dates and (transformed) latent variables         
            smoothed_series = np.zeros(h)
            for t in range(h):
                smoothed_series[t] = self.link(np.dot(Z[self.y.shape[0]+t],a[:,-1]))

            date_index = self.shift_dates(h)

            result = pd.DataFrame(smoothed_series)
            result.rename(columns={0:self.y_name}, inplace=True)
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
            data1 = self.data_original.iloc[:-h+t,:]
            data2 = self.data_original.iloc[-h+t:,:] 
            x = NDynReg(formula=self.formula, data=data1, family=self.family)                                       
            x.fit(print_progress=False)
            if t == 0:
                predictions = x.predict(1,oos_data=data2)
            else:
                predictions = pd.concat([predictions,x.predict(1,oos_data=data2)])
        
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


    def simulation_smoother(self,beta):
        """ Durbin and Koopman simulation smoother - simulates from states 
        given model latent variables and observations

        Parameters
        ----------

        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        - A simulated state evolution
        """         

        T, Z, R, Q = self._ss_matrices(beta)
        H, mu = self._approximating_model(beta,T,Z,R,Q)

        # Generate e_t+ and n_t+
        rnd_h = np.random.normal(0,np.sqrt(H),self.data.shape[0])
        q_dist = ss.multivariate_normal([0.0,0.0], Q,allow_singular=True)
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

    def smoothed_state(self,data,beta,H,mu):
        """ Creates smoothed state estimate given state matrices and 
        parameters.

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
        alpha, V = nld_univariate_KFS(data,Z,H,T,Q,R,mu)
        return alpha, V