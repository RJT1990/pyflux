import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as sp
from patsy import dmatrices, dmatrix, demo_data

from .. import families as fam
from .. import tsm as tsm
from .. import data_check as dc

from .gas_core_recursions import gas_reg_recursion

class GASReg(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GENERALIZED AUTOREGRESSIVE SCORE (GAS) REGRESSION MODELS ****

    Parameters
    ----------

    formula : string
        patsy string describing the regression

    data : pd.DataFrame or np.array
        Field to specify the data that will be used

    family : GAS family object
        Which distribution to use, e.g. GASNormal()

    """

    def __init__(self, formula, data, family):

        # Initialize TSM object     
        super(GASReg,self).__init__('GASReg')

        # Latent Variables
        self.max_lag = 0
        self._z_hide = 0 # Whether to cutoff variance latent variables from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        self.skewness = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.y_name = self.y.design_info.describe()
        self.X_names = self.X.design_info.describe().split(" + ")
        self.y = self.y.astype(np.float) 
        self.X = self.X.astype(np.float) 
        self.z_no = self.X.shape[1]
        self.data_name = self.y_name
        self.y = np.array([self.y]).ravel()
        self.data = self.y
        self.X = np.array([self.X])[0]
        self.index = data.index
        self.initial_values = np.zeros(self.z_no)

        self.data_length = self.data.shape[0]
        self._create_model_matrices()
        self._create_latent_variables()

        self.family = family
        
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, self.cythonized = self.family.setup()
    
        # Identify whether model has cythonized backend - then choose update type
        if self.cythonized is True:
            self._model = self._cythonized_model 
            self._mb_model = self._cythonized_mb_model
            self.recursion = self.family.gradientreg_recursion()
        else:
            self._model = self._uncythonized_model
            self._mb_model = self._uncythonized_mb_model

        self.model_name = self.model_name2 + " GAS Regression"

        # Build any remaining latent variables that are specific to the family chosen
        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0],i[1],i[2])
            self.latent_variables.z_list[no+self.z_no].start = i[3]

        self.family_z_no = len(self.family.build_latent_variables())
        self.z_no += len(self.family.build_latent_variables())

    def _create_model_matrices(self):
        """ Creates model matrices/vectors

        Returns
        ----------
        None (changes model attributes)
        """

        self.model_Y = self.data
        self.model_scores = np.zeros((self.X.shape[1], self.model_Y.shape[0]+1))

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        for parm in range(self.z_no):
            self.latent_variables.add_z('Scale ' + self.X_names[parm], fam.Flat(transform='exp'), fam.Normal(0, 3))
            self.latent_variables.z_list[parm].start = -5.0
        self.z_no = len(self.latent_variables.z_list)

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

    def _cythonized_model(self, beta):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        theta : np.array
            Contains the predicted values for the time series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        scores : np.array
            Contains the scores for the time series
        """

        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        coefficients = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))
        coefficients[:,0] = self.initial_values
        theta = np.zeros(self.model_Y.shape[0]+1)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        theta, self.model_scores, coefficients = self.recursion(parm, theta, self.X, coefficients, self.model_scores, self.model_Y, self.model_Y.shape[0], model_scale, model_shape, model_skewness)

        return np.array(theta[:-1]), self.model_Y, self.model_scores, coefficients

    def _cythonized_mb_model(self, beta, mini_batch):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        mini_batch : int
            Size of each mini batch of data

        Returns
        ----------
        theta : np.array
            Contains the predicted values for the time series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        scores : np.array
            Contains the scores for the time series
        """

        rand_int =  np.random.randint(low=0, high=self.data_length-mini_batch-self.max_lag+1)
        sample = np.arange(start=rand_int, stop=rand_int+mini_batch)

        data = self.data[sample]
        X = self.X[sample, :]
        Y = data[self.max_lag:]

        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        coefficients = np.zeros((X.shape[1], Y.shape[0]+1))
        coefficients[:,0] = self.initial_values
        theta = np.zeros(Y.shape[0]+1)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        theta, self.model_scores, coefficients = self.recursion(parm, theta, X, coefficients, self.model_scores, Y, Y.shape[0], model_scale, model_shape, model_skewness)

        return np.array(theta[:-1]), Y, self.model_scores, coefficients

    def _uncythonized_model(self, beta):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        theta : np.array
            Contains the predicted values for the time series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        scores : np.array
            Contains the scores for the time series
        """

        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        coefficients = np.zeros((self.X.shape[1],self.model_Y.shape[0]+1))
        coefficients[:,0] = self.initial_values
        theta = np.zeros(self.model_Y.shape[0]+1)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        theta, self.model_scores, coefficients = gas_reg_recursion(parm, theta, self.X, coefficients, self.model_scores, self.model_Y, self.model_Y.shape[0], 
            self.family.reg_score_function, self.link, model_scale, model_shape, model_skewness, self.max_lag)

        return theta[:-1], self.model_Y, self.model_scores, coefficients

    def _uncythonized_model(self, beta, mini_batch):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        mini_batch : int
            Size of each mini batch of data

        Returns
        ----------
        theta : np.array
            Contains the predicted values for the time series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)

        scores : np.array
            Contains the scores for the time series
        """

        rand_int =  np.random.randint(low=0, high=self.data_length-mini_batch-self.max_lag+1)
        sample = np.arange(start=rand_int, stop=rand_int+mini_batch)

        data = self.data[sample]
        X = self.X[sample, :]
        Y = data[self.max_lag:]

        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        coefficients = np.zeros((X.shape[1], Y.shape[0]+1))
        coefficients[:,0] = self.initial_values
        theta = np.zeros(Y.shape[0]+1)
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)

        # Loop over time series
        theta, self.model_scores, coefficients = gas_reg_recursion(parm, theta, X, coefficients, self.model_scores, Y, Y.shape[0], 
            self.family.reg_score_function, self.link, model_scale, model_shape, model_skewness, self.max_lag)

        return theta[:-1], Y, self.model_scores, coefficients

    def _preoptimize_model(self, initials, method):
        """ Preoptimizes the model by estimating a static model, then a quick search of good dynamic parameters

        Parameters
        ----------
        initials : np.array
            A vector of inital values

        method : str
            One of 'MLE' or 'PML' (the optimization options)

        Returns
        ----------
        Y_exp : np.array
            Vector of past values and predictions 
        """
        
        # Random search for good starting values
        start_values = []
        start_values.append(np.ones(len(self.X_names))*-2.0)
        start_values.append(np.ones(len(self.X_names))*-3.0)
        start_values.append(np.ones(len(self.X_names))*-4.0)
        start_values.append(np.ones(len(self.X_names))*-5.0)

        best_start = self.latent_variables.get_z_starting_values()
        best_lik = self.neg_loglik(self.latent_variables.get_z_starting_values())
        proposal_start = best_start.copy()

        for start in start_values:
            proposal_start[:len(self.X_names)] = start
            proposal_likelihood = self.neg_loglik(proposal_start)
            if proposal_likelihood < best_lik:
                best_lik = proposal_likelihood
                best_start = proposal_start.copy()

        return best_start

    def neg_loglik(self, beta):
        """ Returns the negative loglikelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables
        """
        theta, Y, scores,_ = self._model(beta)
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y,self.link(theta),model_scale,model_shape,model_skewness)

    def mb_neg_loglik(self, beta, mini_batch):
        """ Returns the negative loglikelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        mini_batch : int
            Size of each mini batch of data
        """
        theta, Y, scores,_ = self._mb_model(beta, mini_batch)
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y,self.link(theta),model_scale,model_shape,model_skewness)

    def plot_fit(self, **kwargs):
        """ Plots the fit of the model

        Notes
        ----------
        Intervals are bootstrapped as follows: take the filtered values from the
        algorithm (thetas). Use these thetas to generate a pseudo data stream from
        the measurement density. Use the GAS algorithm and estimated latent variables to
        filter the pseudo data. Repeat this N times. 

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

            date_index = self.index.copy()
            mu, Y, scores, coefficients = self._model(self.latent_variables.get_z_values())

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

            plt.figure(figsize=figsize) 
            
            plt.subplot(len(self.X_names)+1, 1, 1)
            plt.title(self.y_name + " Filtered")
            plt.plot(date_index,Y,label='Data')
            plt.plot(date_index,values_to_plot,label='GAS Filter',c='black')
            plt.legend(loc=2)

            for coef in range(0,len(self.X_names)):
                plt.subplot(len(self.X_names)+1, 1, 2+coef)
                plt.title("Beta " + self.X_names[coef]) 
                plt.plot(date_index,coefficients[coef,0:-1],label='Coefficient')
                plt.legend(loc=2)               

            plt.show()          
    
    def plot_predict(self, h=5, past_values=20, intervals=True, oos_data=None, **kwargs):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show prediction intervals for the forecast?

        oos_data : pd.DataFrame
            Data for the variables to be used out of sample (ys can be NaNs)

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
            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            X_pred = X_oos[:h]

            date_index = self.shift_dates(h)

            if self.latent_variables.estimation_method in ['M-H']:
                
                sim_vector = np.zeros([15000,h])

                for n in range(0, 15000):
                    t_z = self.draw_latent_variables(nsims=1).T[0]
                    _, Y, _, coefficients = self._model(t_z)
                    coefficients_star = coefficients.T[-1]
                    theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]
                    t_z = np.array([self.latent_variables.z_list[k].prior.transform(t_z[k]) for k in range(t_z.shape[0])])
                    model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)
                    sim_vector[n,:] = self.family.draw_variable(self.link(theta_pred), model_scale, model_shape, model_skewness, theta_pred.shape[0])
                mean_values = np.append(Y, self.link(np.array([np.mean(i) for i in sim_vector.T])))
            else:

                # Retrieve data, dates and (transformed) latent variables
                _, Y, _, coefficients = self._model(self.latent_variables.get_z_values()) 
                coefficients_star = coefficients.T[-1] 
                theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]  
                t_z = self.transform_z()
                sim_vector = np.zeros([15000,h])
                mean_values = np.append(Y, self.link(theta_pred))
                model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)

                if self.model_name2 == "Skewt":
                    m1 = (np.sqrt(model_shape)*sp.gamma((model_shape-1.0)/2.0))/(np.sqrt(np.pi)*sp.gamma(model_shape/2.0))
                    mean_values += (model_skewness - (1.0/model_skewness))*model_scale*m1 

                for n in range(0,15000):
                    sim_vector[n,:] = self.family.draw_variable(self.link(theta_pred),model_scale,model_shape,model_skewness,theta_pred.shape[0])

            sim_vector = sim_vector.T
            error_bars = []
            for pre in range(5,100,5):
                error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector], 0, mean_values[-h-1]))
            forecasted_values = mean_values[-h-1:]
            plot_values = mean_values[-h-past_values:]
            plot_index = date_index[-h-past_values:]

            plt.figure(figsize=figsize)
            if intervals == True:
                alpha =[0.15*i/float(100) for i in range(50,12,-2)]
                for count in range(9):
                    plt.fill_between(date_index[-h-1:], error_bars[count], error_bars[-count],
                        alpha=alpha[count])     
            plt.plot(plot_index,plot_values)
            plt.title("Forecast for " + self.data_name)
            plt.xlabel("Time")
            plt.ylabel(self.data_name)
            plt.show()

    def plot_predict_is(self, h=5, fit_once=True, fit_method='MLE', **kwargs):
        """ Plots forecasts with the estimated model against data
            (Simulated prediction with data)

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
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize', (10,7))

        plt.figure(figsize=figsize)
        predictions = self.predict_is(h=h, fit_method=fit_method, fit_once=fit_once)
        data = self.data[-h:]
        plt.plot(predictions.index,data,label='Data')
        plt.plot(predictions.index,predictions,label='Predictions',c='black')
        plt.title(self.y_name)
        plt.legend(loc=2)   
        plt.show()          

    def predict(self, h=5, oos_data=None, intervals=False, **kwargs):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        oos_data : pd.DataFrame
            Data for the variables to be used out of sample (ys can be NaNs)

        intervals : boolean (default: False)
            Whether to return prediction intervals

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     
        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            # Sort/manipulate the out-of-sample data
            _, X_oos = dmatrices(self.formula, oos_data)
            X_oos = np.array([X_oos])[0]
            X_pred = X_oos[:h]

            date_index = self.shift_dates(h)

            if self.latent_variables.estimation_method in ['M-H']:
                
                sim_vector = np.zeros([15000,h])

                for n in range(0, 15000):
                    t_z = self.draw_latent_variables(nsims=1).T[0]
                    _, Y, _, coefficients = self._model(t_z)
                    coefficients_star = coefficients.T[-1]
                    theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]
                    t_z = np.array([self.latent_variables.z_list[k].prior.transform(t_z[k]) for k in range(t_z.shape[0])])
                    model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)
                    sim_vector[n,:] = self.family.draw_variable(self.link(theta_pred), model_scale, model_shape, model_skewness, theta_pred.shape[0])

                sim_vector = sim_vector.T

                forecasted_values = np.array([np.mean(i) for i in sim_vector])
                prediction_01 = np.array([np.percentile(i, 1) for i in sim_vector])
                prediction_05 = np.array([np.percentile(i, 5) for i in sim_vector])
                prediction_95 = np.array([np.percentile(i, 95) for i in sim_vector])
                prediction_99 = np.array([np.percentile(i, 99) for i in sim_vector])

            else:

                # Retrieve data, dates and (transformed) latent variables
                _, Y, _, coefficients = self._model(self.latent_variables.get_z_values()) 
                coefficients_star = coefficients.T[-1] 
                theta_pred = np.dot(np.array([coefficients_star]), X_pred.T)[0]  
                t_z = self.transform_z()
                mean_values = np.append(Y, self.link(theta_pred))
                model_scale, model_shape, model_skewness = self._get_scale_and_shape(t_z)

                if self.model_name2 == "Skewt":
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
                    sim_values = np.zeros([15000,h])

                    if intervals is True:
                        for n in range(0,15000):
                            sim_values[n,:] = self.family.draw_variable(self.link(theta_pred),model_scale,model_shape,model_skewness,theta_pred.shape[0])

                    sim_values = sim_values.T

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

    def predict_is(self, h=5, fit_once=True, fit_method='MLE', intervals=False):
        """ Makes dynamic in-sample predictions with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        fit_once : boolean
            (default: True) Fits only once before the in-sample prediction; if False, fits after every new datapoint

        fit_method : string
            Which method to fit the model with

        intervals : boolean (default: False)
            Whether to return prediction intervals

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        predictions = []

        for t in range(0,h):
            data1 = self.data_original.iloc[:-h+t,:]
            data2 = self.data_original.iloc[-h+t:,:]
            x = GASReg(formula=self.formula, data=self.data_original[:(-h+t)], family=self.family)

            if fit_once is False:
                x.fit(method=fit_method, printer=False)
            if t == 0:
                if fit_once is True:
                    x.fit(method=fit_method, printer=False)
                    saved_lvs = x.latent_variables
                predictions = x.predict(h=1, oos_data=data2, intervals=intervals)
            else:
                if fit_once is True:
                    x.latent_variables = saved_lvs
                predictions = pd.concat([predictions,x.predict(h=1, oos_data=data2, intervals=intervals)])

        predictions.rename(columns={0:self.y_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

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
            date_index = self.index.copy()
            _, Y, _, _ = self._model(self.latent_variables.get_z_values()) 
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