import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
from patsy import dmatrices, dmatrix, demo_data

from .. import families as fam
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .nn_architecture import neural_network_tanh, neural_network_tanh_mb

import matplotlib.pylab as plt
import seaborn as sns

class NNARX(tsm.TSM):
    """ Inherits time series methods from TSM parent class.

    **** NEURAL NETWORK AUTOREGRESSIVE (NNAR) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Field to specify the univariate time series data that will be used.

    ar : int
        Field to specify how many AR lags the model will have.

    units : int
        How many units for the neural network

    layers : int
        How many layers for the neural networ

    activation : function
        E.g. np.tanh, np.array (linear)

    family: family object
        E.g. pf.Normal()

    """

    def __init__(self, data, formula, ar, units, layers, family=fam.Normal()):

        # Initialize TSM object
        super(NNARX, self).__init__('NNARX')

        # Latent Variable information
        self.ar = ar
        self.units = units
        self.layers = layers
        self.activation = np.tanh
        self.model_name = "NNARX(" + str(self.ar) + ")"
        self.z_no = self.ar + 2
        self.max_lag = self.ar
        self._z_hide = 0 # Whether to cutoff latent variables from results table
        self.supported_methods = ["BBVI"]
        self.default_method = "BBVI"
        self.multivariate_model = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data.copy()
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.y_name = self.y.design_info.describe()
        self.X_names = self.X.design_info.describe().split(" + ")
        self.y = self.y.astype(np.float) 
        self.X = self.X.astype(np.float) 
        self.z_no = self.X.shape[1]
        self.data_name = self.y_name
        self.y = np.array([self.y]).ravel()
        self.data = self.y.copy()
        self.X = np.array([self.X])[0]
        self.index = data.index
        self.data_length = self.data.shape[0]
        self.X = self.X[self.ar:, :]
        self.X = np.concatenate([self._ar_matrix().T, self.X], axis=1).T

        self._create_latent_variables()

        self.family = family
        
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, self.cythonized = self.family.setup()
        
        self.model_name = self.model_name2 + " NNARX(" + str(self.ar) + ")"

        # Build any remaining latent variables that are specific to the family chosen
        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0], i[1], i[2])
            self.latent_variables.z_list[-1].start = i[3]

        self.z_no = len(self.latent_variables.z_list)
        self.family_z_no = len(self.family.build_latent_variables())

        # Initialize with random weights
        for var_no in range(len(self.latent_variables.z_list)-self.family_z_no):
            self.latent_variables.z_list[var_no].start = np.random.normal()

        if isinstance(self.family, fam.Normal):
            self.neg_loglik = self.normal_neg_loglik
        else:
            self.neg_loglik = self.general_neg_loglik

    def _ar_matrix(self):
        """ Creates Autoregressive matrix

        Returns
        ----------
        X : np.ndarray
            Autoregressive Matrix

        """

        if self.ar != 0:
            X = self.data[(self.max_lag-1):-1]
            for i in range(1, self.ar):
                X = np.vstack((X, self.data[(self.max_lag-i-1):-i-1]))
            return X
        else:
            return np.zeros(self.data_length-self.max_lag)
            
    def _create_latent_variables(self):
        """ Creates the model's latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        # Input layer
        for unit in range(self.units):
            self.latent_variables.add_z('Constant | Layer ' + str(1) + ' | Unit ' + str(unit+1), fam.Cauchy(0,1,transform=None), fam.Normal(0, 3))

            for ar_term in range(self.ar):
                self.latent_variables.add_z('AR' + str(ar_term+1) + ' | Layer ' + str(1) + ' | Unit ' + str(unit+1), fam.Cauchy(0,1,transform=None), fam.Normal(0, 3))

            for z in range(len(self.X_names)):
                self.latent_variables.add_z('Weight ' + self.X_names[z], fam.Cauchy(0, 1, transform=None), fam.Normal(0, 3))

        # Hidden layers
        for layer in range(1, self.layers):
            for unit in range(self.units):
                for weight in range(self.units):
                    self.latent_variables.add_z('Weight ' + str(weight+1) + ' | Layer ' + str(layer+1) + ' | Unit ' + str(unit+1), fam.Cauchy(0,1,transform=None), fam.Normal(0, 3))

        # Output layer
        for weight in range(self.units):
            self.latent_variables.add_z('Output Weight ' + str(weight+1), fam.Cauchy(0,1,transform=None), fam.Normal(0, 3))

    def _get_scale_and_shape(self,parm):
        """ Obtains appropriate model scale and shape latent variables

        Parameters
        ----------
        parm : np.array
            Transformed latent variable vector

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

    def _model(self, beta):
        """ Creates the structure of the model (model matrices etc)

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

        return neural_network_tanh(Y, self.X, z, self.units, self.layers, self.ar+len(self.X_names)), Y

    def _mb_model(self, beta, mini_batch):
        """ Creates the structure of the model (model matrices etc) for mini batch model

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

        Y = np.array(self.data[self.max_lag:])

        sample = np.random.choice(len(Y), mini_batch, replace=False)

        Y = Y[sample]
        X = self.X[:, sample]

        # Transform latent variables
        z = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        return neural_network_tanh_mb(Y, X, z, self.units, self.layers, self.ar+len(self.X_names)), Y

    def predict_new(self, X, z):

        first_layer_output = np.zeros(self.units)
        
        for unit in range(self.units):
            first_layer_output[unit] = self.activation(np.matmul(np.transpose(X), z[unit*(self.ar+len(self.X_names)+1):((unit+1)*(self.ar+len(self.X_names)+1))]))

        params_used = ((self.units)*(self.ar+len(self.X_names)+1))

        # Hidden layers
        hidden_layer_output = np.zeros((self.units, self.layers-1))
        for layer in range(1, self.layers):
            for unit in range(self.units):
                if layer == 1:
                    hidden_layer_output[unit,layer-1] = self.activation(np.matmul(first_layer_output,
                        z[params_used+unit*(self.units)+((layer-1)*self.units**2):((params_used+(unit+1)*self.units)+((layer-1)*self.units**2))]))
                else:
                    hidden_layer_output[unit,layer-1] = self.activation(np.matmul(hidden_layer_output[:,layer-1],
                        z[params_used+unit*(self.units)+((layer-1)*self.units**2):((params_used+(unit+1)*self.units)+((layer-1)*self.units**2))]))

        params_used = params_used + (self.layers-1)*self.units**2

        # Output layer
        if self.layers == 1:
            mu = np.matmul(first_layer_output, z[params_used:params_used+self.units])
        else:
            mu = np.matmul(hidden_layer_output[:,-1], z[params_used:params_used+self.units])

        return mu

    def _mean_prediction(self, mu, Y, h, t_z):
        """ Creates a h-step ahead mean prediction

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
        h-length vector of mean predictions
        """     

        # Create arrays to iteratre over
        Y_exp = Y.copy()

        # Loop over h time periods          
        for t in range(0,h):

            if self.ar != 0:
                Y_exp_normalized = (Y_exp[-self.ar:][::-1] - self._norm_mean) / self._norm_std
                new_value = self.predict_new(np.append(1.0, Y_exp_normalized), self.latent_variables.get_z_values())

            else:  
                new_value = self.predict_new(np.array([1.0]), self.latent_variables.get_z_values())

            Y_exp = np.append(Y_exp, [self.link(new_value)])

        return Y_exp

    def _sim_prediction(self, mu, Y, h, t_z, simulations):
        """ Simulates a h-step ahead mean prediction

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

            # Loop over h time periods          
            for t in range(0,h):

                if self.ar != 0:
                    Y_exp_normalized = (Y_exp[-self.ar:][::-1] - self._norm_mean) / self._norm_std
                    new_value = self.predict_new(np.append(1.0, Y_exp_normalized), self.latent_variables.get_z_values())

                else:
                    new_value = self.predict_new(np.array([1.0]), self.latent_variables.get_z_values())

                new_value += np.random.randn(1)*t_z[-1]

                if self.model_name2 == "Exponential":
                    rnd_value = self.family.draw_variable(1.0/self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]
                else:
                    rnd_value = self.family.draw_variable(self.link(new_value), model_scale, model_shape, model_skewness, 1)[0]

                Y_exp = np.append(Y_exp, [rnd_value])

            sim_vector[n] = Y_exp[-h:]

        return np.transpose(sim_vector)

    def _summarize_simulations(self, mean_values, sim_vector, date_index, h, past_values):
        """ Produces simulation forecasted values and prediction intervals

        Parameters
        ----------
        mean_values : np.ndarray
            Mean predictions for h-step ahead forecasts

        sim_vector : np.ndarray
            N simulated predictions for h-step ahead forecasts

        date_index : pd.DateIndex or np.ndarray
            Date index for the simulations

        h : int
            How many steps ahead are forecast

        past_values : int
            How many past observations to include in the forecast plot

        intervals : boolean
            Would you like to show prediction intervals for the forecast?
        """         

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[-h:],0,0))
        forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index
        
    def general_neg_loglik(self, beta):
        """ Calculates the negative log-likelihood of the model

        Parameters
        ----------
        beta : np.ndarray
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._model(beta)
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        #TODO: Replace above with transformation that only acts on scale, shape, skewness in future (speed-up)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y, self.link(mu), model_scale, model_shape, model_skewness)

    def mb_neg_loglik(self, beta, mini_batch):
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
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        #TODO: Replace above with transformation that only acts on scale, shape, skewness in future (speed-up)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y, self.link(mu), model_scale, model_shape, model_skewness)

    def normal_neg_loglik(self, beta):
        """ Creates the negative log-likelihood of the model

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

    def plot_fit(self, **kwargs):
        """ 
        Plots the fit of the model against the data
        """

        ax = kwargs.get('ax', None)
        if ax is None:
            figsize = kwargs.get('figsize', (10, 7))
            fig, ax = plt.subplots(figsize=figsize)
        date_index = self.index[self.ar:self.data.shape[0]]
        mu, Y = self._model(self.latent_variables.get_z_values())
        ax.plot(date_index,Y,label='Data')
        ax.plot(date_index,mu,label='Filter',c='black')
        ax.title(self.data_name)
        ax.legend(loc=2)
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

        ax = kwargs.get('ax', None)
        if ax is None:
            figsize = kwargs.get('figsize', (10, 7))
            fig, ax = plt.subplots(figsize=figsize)

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:


            # Retrieve data, dates and (transformed) latent variables
            mu, Y = self._model(self.latent_variables.get_z_values())         
            date_index = self.shift_dates(h)
            t_z = self.transform_z()

            # Get mean prediction and simulations (for errors)
            mean_values = self._mean_prediction(mu, Y, h, t_z)
            if intervals is True:
                sim_values = self._sim_prediction(mu, Y, h, t_z, 15000)
            else:
                sim_values = self._sim_prediction(mu, Y, h, t_z, 2)
            error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(mean_values, sim_values, date_index, h, past_values)

            if intervals is True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count, pre in enumerate(error_bars):
                    plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])            
            ax.plot(plot_index,plot_values)
            ax.title("Forecast for " + self.data_name)
            ax.xlabel("Time")
            ax.ylabel(self.data_name)
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
            x = NNAR(ar=self.ar, units=self.units, 
                layers=self.layers, data=self.data_original[:-h+t], family=self.family)
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
        """ Plots forecasts with the estimated model against data (Simulated prediction with data)

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
        - Plot of the forecast against data 
        """     


        ax = kwargs.get('ax', None)
        if ax is None:
            figsize = kwargs.get('figsize', (10, 7))
            fig, ax = plt.subplots(figsize=figsize)
        predictions = self.predict_is(h, fit_method=fit_method, fit_once=fit_once)
        data = self.data[-h:]
        ax.plot(predictions.index, data, label='Data')
        ax.plot(predictions.index, predictions, label='Predictions', c='black')
        ax.title(self.data_name)
        ax.legend(loc=2)
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
            ax = kwargs.get('ax', None)
            if ax is None:
                figsize = kwargs.get('figsize', (10, 7))
                fig, ax = plt.subplots(figsize=figsize)
            date_index = self.index[self.ar:self.data_length]
            mu, Y = self._model(self.latent_variables.get_z_values())
            draws = self.sample(nsims).T
            ax.plot(date_index, draws, label='Posterior Draws', alpha=1.0)
            if plot_data is True:
                ax.plot(date_index, Y, label='Data', c='black', alpha=0.5, linestyle='', marker='s')
            ax.title(self.data_name)
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

            ax = kwargs.get('ax', None)
            if ax is None:
                figsize = kwargs.get('figsize', (10, 7))
                fig, ax = plt.subplots(figsize=figsize)

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

            ax = plt.subplot()
            ax.axvline(T_actual)
            sns.distplot(T_sim, kde=False, ax=ax)
            ax.set(title='Posterior predictive' + description, xlabel='T(x)', ylabel='Frequency');
            plt.show()