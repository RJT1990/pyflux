import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as sp

from .. import families as fam
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .arma_recursions import arima_recursion, arima_recursion_normal, arima_recursion_poisson

class ARIMA(tsm.TSM):
    """ Inherits time series methods from the TSM parent class.

    **** AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Field to specify the univariate time series data that will be used.

    ar : int
        Field to specify how many AR lags the model will have.

    ma : int
        Field to specify how many MA lags the model will have.

    integ : int (default : 0)
        Specifies how many times to difference the time series.

    target : str (if data is a pd.DataFrame) or int (if data is a np.ndarray)
        Specifies which column name or array index to use. By default, first
        column/array index will be selected as the dependent variable.

    family : family object
        E.g pf.Normal(), pf.t(), pf.Laplace()...
    """

    def __init__(self, data, ar, ma, integ=0, target=None, family=fam.Normal()):

        super(ARIMA, self).__init__('ARIMA')

        # Latent Variable information
        self.ar = ar
        self.ma = ma
        self.integ = integ
        self.z_no = self.ar + self.ma + 2
        self.max_lag = max(self.ar,self.ma)
        self._z_hide = 0 # Whether to cutoff latent variables from results table
        self.supported_methods = ["MLE", "PML", "Laplace", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data,target)
        self.data = self.data.astype(np.float) # treat as float for Cython
        self.data_original = self.data.copy()

        # Difference data
        for order in range(0, self.integ):
            self.data = np.diff(self.data)
            self.data_name = "Differenced " + self.data_name
        self.data_length = self.data.shape[0]

        self.X = self._ar_matrix()
        self._create_latent_variables()

        self.family = family
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, self.cythonized = self.family.setup()
        self.model_name = self.model_name2 + " ARIMA(" + str(self.ar) + "," + str(self.integ) + "," + str(self.ma) + ")"

        # Build any remaining latent variables that are specific to the family chosen
        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0], i[1], i[2])
            self.latent_variables.z_list[1+no+self.ar+self.ma].start = i[3]
        self.latent_variables.z_list[0].start = self.mean_transform(np.mean(self.data))

        self.family_z_no = len(self.family.build_latent_variables())
        self.z_no = len(self.latent_variables.z_list)

        # If Normal family is selected, we use faster likelihood functions
        if isinstance(self.family, fam.Normal):
            self._model = self._normal_model
            self._mb_model = self._mb_normal_model
            self.neg_loglik = self.normal_neg_loglik
            self.mb_neg_loglik = self.normal_mb_neg_loglik
        elif isinstance(self.family, fam.t) or isinstance(self.family, fam.Cauchy) or isinstance(self.family, fam.Laplace):
            self._model = self._normal_model
            self._mb_model = self._mb_normal_model
            self.neg_loglik = self.non_normal_neg_loglik
            self.mb_neg_loglik = self.non_normal_mb_neg_loglik
        elif isinstance(self.family, fam.Poisson):
            self._model = self._poisson_model
            self._mb_model = self._mb_poisson_model
            self.neg_loglik = self.non_normal_neg_loglik
            self.mb_neg_loglik = self.non_normal_mb_neg_loglik    
        else:
            self._model = self._non_normal_model
            self._mb_model = self._mb_non_normal_model
            self.neg_loglik = self.non_normal_neg_loglik
            self.mb_neg_loglik = self.non_normal_mb_neg_loglik

    def _ar_matrix(self):
        """ Creates the Autoregressive matrix for the model

        Returns
        ----------
        X : np.ndarray
            Autoregressive Matrix
        """

        X = np.ones(self.data_length-self.max_lag)

        if self.ar != 0:
            for i in range(0, self.ar):
                X = np.vstack((X,self.data[(self.max_lag-i-1):-i-1]))

        return X

    def _create_latent_variables(self):
        """ Creates the model's latent variables

        A latent variable addition requires:

        - Latent variable name - e.g Constant
        - Latent variable prior family - e.g. Normal(0, 1)
        - Variational approximation - e.g. Normal(0, 1)

        Returns
        ----------
        None (adds to self.latent_variables)
        """

        self.latent_variables.add_z('Constant', fam.Normal(0,3,transform=None), fam.Normal(0, 3))

        for ar_term in range(self.ar):
            self.latent_variables.add_z('AR(' + str(ar_term+1) + ')', fam.Normal(0,0.5,transform=None), fam.Normal(0, 3))

        for ma_term in range(self.ma):
            self.latent_variables.add_z('MA(' + str(ma_term+1) + ')', fam.Normal(0,0.5,transform=None), fam.Normal(0, 3))

    def _get_scale_and_shape(self, transformed_lvs):
        """ Obtains model scale, shape and skewness latent variables

        Parameters
        ----------
        transformed_lvs : np.array
            Transformed latent variable vector

        Returns
        ----------
        - Tuple of model scale, model shape, model skewness
        """

        if self.scale is True:
            if self.shape is True:
                model_shape = transformed_lvs[-1]  
                model_scale = transformed_lvs[-2]
            else:
                model_shape = 0
                model_scale = transformed_lvs[-1]
        else:
            model_scale = 0
            model_shape = 0 

        if self.skewness is True:
            model_skewness = transformed_lvs[-3]
        else:
            model_skewness = 0

        return model_scale, model_shape, model_skewness

    def _get_scale_and_shape_sim(self, transformed_lvs):
        """ Obtains model scale, shape, skewness latent variables for
        a 2d array of simulations.

        Parameters
        ----------
        transformed_lvs : np.array
            Transformed latent variable vector (2d - with draws of each variable)

        Returns
        ----------
        - Tuple of np.arrays (each being scale, shape and skewness draws)
        """

        if self.scale is True:
            if self.shape is True:
                model_shape = self.latent_variables.z_list[-1].prior.transform(transformed_lvs[-1, :]) 
                model_scale = self.latent_variables.z_list[-2].prior.transform(transformed_lvs[-2, :])
            else:
                model_shape = np.zeros(transformed_lvs.shape[1])
                model_scale = self.latent_variables.z_list[-1].prior.transform(transformed_lvs[-1, :])
        else:
            model_scale = np.zeros(transformed_lvs.shape[1])
            model_shape = np.zeros(transformed_lvs.shape[1])

        if self.skewness is True:
            model_skewness = self.latent_variables.z_list[-3].prior.transform(transformed_lvs[-3, :])
        else:
            model_skewness = np.zeros(transformed_lvs.shape[1])

        return model_scale, model_shape, model_skewness

    def _normal_model(self, beta):
        """ Creates the structure of the model (model matrices etc) for
        a Normal family ARIMA model.

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        Y = np.array(self.data[self.max_lag:])

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(self.X),z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion_normal(z, mu, Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _poisson_model(self, beta):
        """ Creates the structure of the model (model matrices etc) for
        a Poisson model. Here we apply a link function to the MA lags.

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        Y = np.array(self.data[self.max_lag:])

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            if self.family_z_no + self.ma == 0:
                mu = np.matmul(np.transpose(self.X), z)
            else:
                mu = np.matmul(np.transpose(self.X), z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion_poisson(z, mu, self.link(mu), Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _non_normal_model(self, beta):
        """ Creates the structure of the model (model matrices etc) for
        a non-normal model. Here we apply a link function to the MA lags.

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        Y = np.array(self.data[self.max_lag:])

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(self.X),z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion(z, mu, self.link(mu), Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _mb_normal_model(self, beta, mini_batch):
        """ Creates the structure of the model (model matrices etc) for mini batch model.
        
        Here the structure is the same as for _normal_model() but we are going to
        sample a random choice of data points (of length mini_batch).

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        mini_batch : int
            Mini batch size for the data sampling

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        rand_int =  np.random.randint(low=0, high=self.data.shape[0]-mini_batch-self.max_lag+1)
        sample = np.arange(start=rand_int, stop=rand_int+mini_batch)

        Y = self.data[self.max_lag:][sample]
        X = self.X[:, sample]

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(X),z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion_normal(z, mu, Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _mb_non_normal_model(self, beta, mini_batch):
        """ Creates the structure of the model (model matrices etc) for mini batch model

        Here the structure is the same as for _non_normal_model() but we are going to
        sample a random choice of data points (of length mini_batch).

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        mini_batch : int
            Mini batch size for the data sampling

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        rand_int =  np.random.randint(low=0, high=self.data_length-mini_batch-self.max_lag+1)
        sample = np.arange(start=rand_int, stop=rand_int+mini_batch)

        Y = self.data[self.max_lag:][sample]
        X = self.X[:, sample]

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(X),z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion(z, mu, self.link(mu), Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 


    def _mb_poisson_model(self, beta, mini_batch):
        """ Creates the structure of the model (model matrices etc) for mini batch model

        Here the structure is the same as for _non_normal_model() but we are going to
        sample a random choice of data points (of length mini_batch).

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for the latent variables

        mini_batch : int
            Mini batch size for the data sampling

        Returns
        ----------
        mu : np.ndarray
            Contains the predicted values (location) for the time series

        Y : np.ndarray
            Contains the length-adjusted time series (accounting for lags)
        """     

        rand_int =  np.random.randint(low=0, high=self.data_length-mini_batch-self.max_lag+1)
        sample = np.arange(start=rand_int, stop=rand_int+mini_batch)

        Y = self.data[self.max_lag:][sample]
        X = self.X[:, sample]

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        # Constant and AR terms
        if self.ar != 0:
            mu = np.matmul(np.transpose(X),z[:-self.family_z_no-self.ma])
        else:
            mu = np.ones(Y.shape[0])*z[0]
            
        # MA terms
        if self.ma != 0:
            mu = arima_recursion_poisson(z, mu, self.link(mu), Y, self.max_lag, Y.shape[0], self.ar, self.ma)

        return mu, Y 

    def _mean_prediction(self, mu, Y, h, t_z):
        """ Creates a h-step ahead mean prediction

        This function is used for predict(). We have to iterate over the number
        of timepoints (h) that the user wants to predict, using as inputs the ARIMA
        parameters, past datapoints, and past predicted datapoints.

        Parameters
        ----------
        mu : np.ndarray
            The past predicted values

        Y : np.ndarray
            The past data

        h : int
            How many steps ahead for the prediction

        t_z : np.ndarray
            A vector of (transformed) latent variables

        Returns
        ----------
        h-length np.array of mean predictions
        """     

        # Create arrays to iteratre over
        Y_exp = Y.copy()
        mu_exp = mu.copy()

        # Loop over h time periods          
        for t in range(0,h):
            new_value = t_z[0]

            if self.ar != 0:
                for j in range(1, self.ar+1):
                    new_value += t_z[j]*Y_exp[-j]

            if self.ma != 0:
                for k in range(1, self.ma+1):
                    if (k-1) >= t:
                        new_value += t_z[k+self.ar]*(Y_exp[-k]-self.link(mu_exp[-k]))

            if self.model_name2 == "Exponential":
                Y_exp = np.append(Y_exp, [1.0/self.link(new_value)])
            else:
                Y_exp = np.append(Y_exp, [self.link(new_value)])

            mu_exp = np.append(mu_exp,[0]) # For indexing consistency

        return Y_exp

    def _sim_prediction(self, mu, Y, h, t_z, simulations):
        """ Simulates a h-step ahead mean prediction

        Same as _mean_prediction() but now we repeat the process 
        by a number of times (simulations) and shock the process
        with random draws from the family, e.g. Normal shocks.

        Parameters
        ----------
        mu : np.ndarray
            The past predicted values

        Y : np.ndarray
            The past data

        h : int
            How many steps ahead for the prediction

        t_params : np.ndarray
            A vector of (transformed) latent variables

        simulations : int
            How many simulations to perform

        Returns
        ----------
        Matrix of simulations
        """     

        model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)

        sim_vector = np.zeros([simulations,h])

        for n in range(0, simulations):
            # Create arrays to iteratre over        
            Y_exp = Y.copy()
            mu_exp = mu.copy()

            # Loop over h time periods          
            for t in range(0,h):

                new_value = t_z[0]

                if self.ar != 0:
                    for j in range(1, self.ar+1):
                        new_value += t_z[j]*Y_exp[-j]

                if self.ma != 0:
                    for k in range(1, self.ma+1):
                        if (k-1) >= t:
                            new_value += t_z[k+self.ar]*(Y_exp[-k]-mu_exp[-k])

                if self.model_name2 == "Exponential":
                    rnd_value = self.family.draw_variable(1.0/self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]
                else:
                    rnd_value = self.family.draw_variable(self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]

                Y_exp = np.append(Y_exp, [rnd_value])
                mu_exp = np.append(mu_exp, [0]) # For indexing consistency

                sim_vector[n] = Y_exp[-h:]

        return np.transpose(sim_vector)

    def _sim_prediction_bayes(self, h, simulations):
        """ Simulates a h-step ahead mean prediction

        Same as _mean_prediction() but now we repeat the process 
        by a number of times (simulations) and shock the process
        with random draws from the family, e.g. Normal shocks.

        Parameters
        ----------
        h : int
            How many steps ahead for the prediction

        simulations : int
            How many simulations to perform

        Returns
        ----------
        Matrix of simulations
        """     

        sim_vector = np.zeros([simulations,h])

        for n in range(0, simulations):

            t_z = self.draw_latent_variables(nsims=1).T[0]
            mu, Y = self._model(t_z)  
            t_z = np.array([self.latent_variables.z_list[k].prior.transform(t_z[k]) for k in range(t_z.shape[0])])

            model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)

            # Create arrays to iteratre over        
            Y_exp = Y.copy()
            mu_exp = mu.copy()

            # Loop over h time periods          
            for t in range(0,h):

                new_value = t_z[0]

                if self.ar != 0:
                    for j in range(1, self.ar+1):
                        new_value += t_z[j]*Y_exp[-j]

                if self.ma != 0:
                    for k in range(1, self.ma+1):
                        if (k-1) >= t:
                            new_value += t_z[k+self.ar]*(Y_exp[-k]-mu_exp[-k])

                if self.model_name2 == "Exponential":
                    rnd_value = self.family.draw_variable(1.0/self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]
                else:
                    rnd_value = self.family.draw_variable(self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]

                Y_exp = np.append(Y_exp, [rnd_value])
                mu_exp = np.append(mu_exp, [0]) # For indexing consistency

                sim_vector[n] = Y_exp[-h:]

        return np.transpose(sim_vector)

    def _summarize_simulations(self, mean_values, sim_vector, date_index, h, past_values):
        """ Produces simulation forecasted values and prediction intervals. 

        This is a utility function that constructs the prediction intervals and other quantities 
        used for plot_predict() in particular.

        Parameters
        ----------
        mean_values : np.ndarray
            Mean predictions for h-step ahead forecasts

        sim_vector : np.ndarray
            N simulated predictions for h-step ahead forecasts

        date_index : pd.DateIndex or np.ndarray
            Date index for the simulations

        h : int
            How many steps ahead to forecast

        past_values : int
            How many past observations to include in the forecast plot
        """         

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector], 0, mean_values[-h-1]))
        if self.latent_variables.estimation_method in ['M-H']:
            forecasted_values = np.insert([np.mean(i) for i in sim_vector], 0, mean_values[-h-1])
        else:
            forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index
        
    def normal_neg_loglik(self, beta):
        """ Calculates the negative log-likelihood of the model for Normal family

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._model(beta)
        return -np.sum(ss.norm.logpdf(Y, loc=mu, scale=self.latent_variables.z_list[-1].prior.transform(beta[-1])))

    def normal_mb_neg_loglik(self, beta, mini_batch):
        """ Calculates the negative log-likelihood of the Normal model for a minibatch

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        mini_batch : int
            Size of each mini batch of data

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._mb_model(beta, mini_batch)
        return -np.sum(ss.norm.logpdf(Y, loc=mu, scale=self.latent_variables.z_list[-1].prior.transform(beta[-1])))

    def non_normal_neg_loglik(self, beta):
        """ Calculates the negative log-likelihood of the model for a non-Normal family

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._model(beta)
        transformed_parameters = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(transformed_parameters)
        return self.family.neg_loglikelihood(Y, self.link(mu), model_scale, model_shape, model_skewness)

    def non_normal_mb_neg_loglik(self, beta, mini_batch):
        """ Calculates the negative log-likelihood of the model for a minibatch

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        mini_batch : int
            Size of each mini batch of data

        Returns
        ----------
        The negative logliklihood of the model
        """     
        mu, Y = self._mb_model(beta, mini_batch)
        transformed_parameters = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(transformed_parameters)
        return self.family.neg_loglikelihood(Y, self.link(mu), model_scale, model_shape, model_skewness)

    def plot_fit(self, **kwargs):
        """ 
        Plots the fit of the model against the data
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))
        plt.figure(figsize=figsize)
        date_index = self.index[max(self.ar, self.ma):self.data_length]
        mu, Y = self._model(self.latent_variables.get_z_values())

        # Catch specific family properties (imply different link functions/moments)
        if self.model_name2 == "Exponential":
            values_to_plot = 1.0/self.link(mu)
        elif self.model_name2 == "Skewt":
            t_params = self.transform_z()
            model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_params)
            m1 = (np.sqrt(model_shape)*sp.gamma((model_shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(model_shape/2.0))
            additional_loc = (model_skewness - (1.0/model_skewness))*model_scale*m1
            values_to_plot = mu + additional_loc
        else:
            values_to_plot = self.link(mu)

        plt.plot(date_index, Y, label='Data')
        plt.plot(date_index, values_to_plot, label='ARIMA model', c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def plot_predict(self, h=5, past_values=20, intervals=True, **kwargs):
        """ Plots forecasts with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : boolean
            Would you like to show prediction intervals for the forecast?

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

            mu, Y = self._model(self.latent_variables.get_z_values())   
            date_index = self.shift_dates(h)

            if self.latent_variables.estimation_method in ['M-H']:
                sim_vector = self._sim_prediction_bayes(h, 15000)
                error_bars = []

                for pre in range(5,100,5):
                    error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector], 0, Y[-1]))

                forecasted_values = np.insert([np.mean(i) for i in sim_vector], 0, Y[-1])
                plot_values = np.append(Y[-1-past_values:-2], forecasted_values)
                plot_index = date_index[-h-past_values:]

            else:
                t_z = self.transform_z()
                mean_values = self._mean_prediction(mu, Y, h, t_z)

                if self.model_name2 == "Skewt":
                    model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)
                    m1 = (np.sqrt(model_shape)*sp.gamma((model_shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(model_shape/2.0))
                    forecasted_values = mean_values[-h:] + (model_skewness - (1.0/model_skewness))*model_scale*m1 
                else:
                    forecasted_values = mean_values[-h:] 

                if intervals is True:
                    sim_values = self._sim_prediction(mu, Y, h, t_z, 15000)
                else:
                    sim_values = self._sim_prediction(mu, Y, h, t_z, 2)

                error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values, sim_values, date_index, h, past_values)

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], error_bars[count], error_bars[-count-1],alpha=alpha[count])             
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
            plt.show()

    def predict_is(self, h=5, fit_once=True, fit_method='MLE', intervals=False, **kwargs):
        """ Makes dynamic out-of-sample predictions with the estimated model on in-sample data

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        fit_once : boolean
            (default: True) Fits only once before the in-sample prediction; if False, fits after every new datapoint

        fit_method : string
            Which method to fit the model with

        intervals: boolean
            Whether to return prediction intervals

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     
        predictions = []

        for t in range(0,h):
            x = ARIMA(ar=self.ar, ma=self.ma, integ=self.integ, data=self.data_original[:-h+t], family=self.family)
            if fit_once is False:
                x.fit(method=fit_method, printer=False)
            if t == 0:
                if fit_once is True:
                    x.fit(method=fit_method, printer=False)
                    saved_lvs = x.latent_variables
                predictions = x.predict(1, intervals=intervals)
            else:
                if fit_once is True:
                    x.latent_variables = saved_lvs
                predictions = pd.concat([predictions,x.predict(1, intervals=intervals)])
        
        if intervals is True:
            predictions.rename(columns={0:self.data_name, 1: "1% Prediction Interval", 
                2: "5% Prediction Interval", 3: "95% Prediction Interval", 4: "99% Prediction Interval"}, inplace=True)
        else:
            predictions.rename(columns={0:self.data_name}, inplace=True)

        predictions.index = self.index[-h:]

        return predictions

    def plot_predict_is(self, h=5, fit_once=True, fit_method='MLE', **kwargs):
        """ Plots forecasts with the estimated model against data

        Parameters
        ----------
        h : int (default : 5)
            How many steps to forecast

        fit_once : boolean
            (default: True) Fits only once before the in-sample prediction; if False, fits after every new datapoint

        fit_method : string
            Which method to fit the model with

        Returns
        ----------
        - Plot of the forecast against data in-sample
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))
        plt.figure(figsize=figsize)
        predictions = self.predict_is(h, fit_method=fit_method, fit_once=fit_once)
        data = self.data[-h:]
        plt.plot(predictions.index, data, label='Data')
        plt.plot(predictions.index, predictions, label='Predictions', c='black')
        plt.title(self.data_name)
        plt.legend(loc=2)   
        plt.show()          

    def predict(self, h=5, intervals=False):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        intervals : boolean (default: False)
            Whether to return prediction intervals

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:

            mu, Y = self._model(self.latent_variables.get_z_values())   
            date_index = self.shift_dates(h)

            if self.latent_variables.estimation_method in ['M-H']:
                sim_vector = self._sim_prediction_bayes(h, 15000)

                forecasted_values = np.array([np.mean(i) for i in sim_vector])
                prediction_01 = np.array([np.percentile(i, 1) for i in sim_vector])
                prediction_05 = np.array([np.percentile(i, 5) for i in sim_vector])
                prediction_95 = np.array([np.percentile(i, 95) for i in sim_vector])
                prediction_99 = np.array([np.percentile(i, 99) for i in sim_vector])

            else:
                t_z = self.transform_z()
                mean_values = self._mean_prediction(mu, Y, h, t_z)
                if intervals is True:
                    sim_values = self._sim_prediction(mu, Y, h, t_z, 15000)
                else:
                    sim_values = self._sim_prediction(mu, Y, h, t_z, 2)

                if self.model_name2 == "Skewt":
                    model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)
                    m1 = (np.sqrt(model_shape)*sp.gamma((model_shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(model_shape/2.0))
                    forecasted_values = mean_values[-h:] + (model_skewness - (1.0/model_skewness))*model_scale*m1 
                else:
                    forecasted_values = mean_values[-h:] 

            if intervals is False:
                result = pd.DataFrame(forecasted_values)
                result.rename(columns={0:self.data_name}, inplace=True)
            else:
                # Get mean prediction and simulations (for errors)
                if self.latent_variables.estimation_method not in ['M-H']:
                    sim_values = self._sim_prediction(mu, Y, h, t_z, 15000)
                    prediction_01 = np.array([np.percentile(i, 1) for i in sim_values])
                    prediction_05 = np.array([np.percentile(i, 5) for i in sim_values])
                    prediction_95 = np.array([np.percentile(i, 95) for i in sim_values])
                    prediction_99 = np.array([np.percentile(i, 99) for i in sim_values])

                result = pd.DataFrame([forecasted_values, prediction_01, prediction_05, 
                    prediction_95, prediction_99]).T
                result.rename(columns={0:self.data_name, 1: "1% Prediction Interval", 
                    2: "5% Prediction Interval", 3: "95% Prediction Interval", 4: "99% Prediction Interval"}, 
                    inplace=True)
 
            result.index = date_index[-h:]

            return result

    def sample(self, nsims=1000):
        """ Samples from the posterior predictive distribution

        Parameters
        ----------
        nsims : int (default : 1000)
            How many draws from the posterior predictive distribution

        Returns
        ----------
        - np.ndarray of draws from the data
        """     
        if self.latent_variables.estimation_method not in ['BBVI', 'M-H']:
            raise Exception("No latent variables estimated!")
        else:
            lv_draws = self.draw_latent_variables(nsims=nsims)
            mus = [self._model(lv_draws[:,i])[0] for i in range(nsims)]
            model_scale, model_shape, model_skewness = self._get_scale_and_shape_sim(lv_draws)
            data_draws = np.array([self.family.draw_variable(self.link(mus[i]), 
                np.repeat(model_scale[i], mus[i].shape[0]), np.repeat(model_shape[i], mus[i].shape[0]), 
                np.repeat(model_skewness[i], mus[i].shape[0]), mus[i].shape[0]) for i in range(nsims)])
            return data_draws

    def plot_sample(self, nsims=10, plot_data=True, **kwargs):
        """
        Plots draws from the posterior predictive density against the data

        Parameters
        ----------
        nsims : int (default : 1000)
            How many draws from the posterior predictive distribution

        plot_data boolean
            Whether to plot the data or not
        """

        if self.latent_variables.estimation_method not in ['BBVI', 'M-H']:
            raise Exception("No latent variables estimated!")
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns

            figsize = kwargs.get('figsize',(10,7))
            plt.figure(figsize=figsize)
            date_index = self.index[max(self.ar, self.ma):self.data_length]
            mu, Y = self._model(self.latent_variables.get_z_values())
            draws = self.sample(nsims).T
            plt.plot(date_index, draws, label='Posterior Draws', alpha=1.0)
            if plot_data is True:
                plt.plot(date_index, Y, label='Data', c='black', alpha=0.5, linestyle='', marker='s')
            plt.title(self.data_name)
            plt.show()    

    def ppc(self, nsims=1000, T=np.mean):
        """ Computes posterior predictive p-value

        Parameters
        ----------
        nsims : int (default : 1000)
            How many draws for the PPC

        T : function
            A discrepancy measure - e.g. np.mean, np.std, np.max

        Returns
        ----------
        - float (posterior predictive p-value)
        """     
        if self.latent_variables.estimation_method not in ['BBVI', 'M-H']:
            raise Exception("No latent variables estimated!")
        else:
            lv_draws = self.draw_latent_variables(nsims=nsims)
            mus = [self._model(lv_draws[:,i])[0] for i in range(nsims)]
            model_scale, model_shape, model_skewness = self._get_scale_and_shape_sim(lv_draws)
            data_draws = np.array([self.family.draw_variable(self.link(mus[i]), 
                np.repeat(model_scale[i], mus[i].shape[0]), np.repeat(model_shape[i], mus[i].shape[0]), 
                np.repeat(model_skewness[i], mus[i].shape[0]), mus[i].shape[0]) for i in range(nsims)])
            T_sims = T(self.sample(nsims=nsims), axis=1)
            T_actual = T(self.data)
            return len(T_sims[T_sims>T_actual])/nsims

    def plot_ppc(self, nsims=1000, T=np.mean, **kwargs):
        """ Plots histogram of the discrepancy from draws of the posterior

        Parameters
        ----------
        nsims : int (default : 1000)
            How many draws for the PPC

        T : function
            A discrepancy measure - e.g. np.mean, np.std, np.max
        """     
        if self.latent_variables.estimation_method not in ['BBVI', 'M-H']:
            raise Exception("No latent variables estimated!")
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns

            figsize = kwargs.get('figsize',(10,7))

            lv_draws = self.draw_latent_variables(nsims=nsims)
            mus = [self._model(lv_draws[:,i])[0] for i in range(nsims)]
            model_scale, model_shape, model_skewness = self._get_scale_and_shape_sim(lv_draws)
            data_draws = np.array([self.family.draw_variable(self.link(mus[i]), 
                np.repeat(model_scale[i], mus[i].shape[0]), np.repeat(model_shape[i], mus[i].shape[0]), 
                np.repeat(model_skewness[i], mus[i].shape[0]), mus[i].shape[0]) for i in range(nsims)])
            T_sim = T(self.sample(nsims=nsims), axis=1)
            T_actual = T(self.data)

            if T == np.mean:
                description = " of the mean"
            elif T == np.max:
                description = " of the maximum"
            elif T == np.min:
                description = " of the minimum"
            elif T == np.median:
                description = " of the median"
            else:
                description = ""

            plt.figure(figsize=figsize)
            ax = plt.subplot()
            ax.axvline(T_actual)
            sns.distplot(T_sim, kde=False, ax=ax)
            ax.set(title='Posterior predictive' + description, xlabel='T(x)', ylabel='Frequency');
            plt.show()