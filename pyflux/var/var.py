import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import datetime

from .. import families as fam
from .. import output as op
from .. import tests as tst
from .. import tsm as tsm
from .. import data_check as dc

from .var_recursions import create_design_matrix, custom_covariance_matrix, var_likelihood

def create_design_matrix_2(Z, data, Y_len, lag_no):
    """
    For Python 2.7 - cythonized version only works for 3.5
    """
    row_count = 1

    for lag in range(1, lag_no+1):
        for reg in range(Y_len):
            Z[row_count, :] = data[reg][(lag_no-lag):-lag]
            row_count += 1

    return Z


class VAR(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** VECTOR AUTOREGRESSION (VAR) MODELS ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    lags : int
        Field to specify how many lag terms the model will have. 

    integ : int (default : 0)
        Specifies how many time to difference the dependent variables.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.

    use_ols_covariance : Boolean
        If true use OLS covariance; if false use estimated covariance
    """

    def __init__(self,data,lags,target=None,integ=0,use_ols_covariance=False):

        # Initialize TSM object     
        super(VAR,self).__init__('VAR')

        self.neg_logposterior = self.multivariate_neg_logposterior

        # Use uncythonized version of this function for older Pythons
        if sys.version_info < (3,):
            self.create_design_matrix = create_design_matrix_2
        else:
            self.create_design_matrix = create_design_matrix

        # Latent Variables
        self.lags = lags
        self.integ = integ
        self.max_lag = lags
        self.model_name = "VAR(" + str(self.lags) + ")"
        self.supported_methods = ["OLS","MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "OLS"
        self.multivariate_model = True
        self.use_ols_covariance = use_ols_covariance

        # Format the data
        self.data_original = data.copy()
        self.data, self.data_name, self.is_pandas, self.index = dc.mv_data_check(data,target)

        # Difference data
        X = np.transpose(self.data)
        for order in range(self.integ):
            X = np.asarray([np.diff(i) for i in X])
            self.data_name = np.asarray(["Differenced " + str(i) for i in self.data_name])
        self.data = X   
        self.ylen = self.data_name.shape[0]

        self._create_latent_variables()

        # Other attributes
        self._z_hide = np.power(self.data.shape[0],2) - (np.power(self.data.shape[0],2) - self.data.shape[0])/2 # Whether to cutoff variance latent variables from results        
        self.z_no = len(self.latent_variables.z_list)

    def _create_B(self,Y):
        """ Creates OLS coefficient matrix

        Parameters
        ----------
        Y : np.array
            The dependent variables Y

        Returns
        ----------
        The coefficient matrix B
        """         

        Z = self._create_Z(Y)
        return np.dot(np.dot(Y,np.transpose(Z)),np.linalg.inv(np.dot(Z,np.transpose(Z))))

    def _create_B_direct(self):
        """ Creates OLS coefficient matrix (calculates Y within - for OLS fitting)

        Returns
        ----------
        The coefficient matrix B
        """         

        Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])        
        Z = self._create_Z(Y)
        return np.dot(np.dot(Y,np.transpose(Z)),np.linalg.inv(np.dot(Z,np.transpose(Z))))

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        # TODO: There must be a cleaner way to do this below

        # Create VAR latent variables
        for variable in range(self.ylen):
            self.latent_variables.add_z(self.data_name[variable] + ' Constant', fam.Normal(0,3,transform=None), fam.Normal(0,3))
            other_variables = np.delete(range(self.ylen), [variable])
            for lag_no in range(self.lags):
                self.latent_variables.add_z(str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', fam.Normal(0,0.5,transform=None), fam.Normal(0,3))
                for other in other_variables:
                    self.latent_variables.add_z(str(self.data_name[other]) + ' to ' + str(self.data_name[variable]) + ' AR(' + str(lag_no+1) + ')', fam.Normal(0,0.5,transform=None), fam.Normal(0,3))

        starting_params_temp = self._create_B_direct().flatten()

        # Variance latent variables
        for i in range(self.ylen):
            for k in range(self.ylen):
                if i == k:
                    self.latent_variables.add_z('Cholesky Diagonal ' + str(i), fam.Flat(transform='exp'), fam.Normal(0,3))
                elif i > k:
                    self.latent_variables.add_z('Cholesky Off-Diagonal (' + str(i) + ',' + str(k) + ')', fam.Flat(transform=None), fam.Normal(0,3))

        for i in range(0,self.ylen):
            for k in range(0,self.ylen):
                if i == k:
                    starting_params_temp = np.append(starting_params_temp,np.array([0.5]))
                elif i > k:
                    starting_params_temp = np.append(starting_params_temp,np.array([0.0]))

        self.latent_variables.set_z_starting_values(starting_params_temp)

    def _create_Z(self,Y):
        """ Creates design matrix holding the lagged variables

        Parameters
        ----------
        Y : np.array
            The dependent variables Y

        Returns
        ----------
        The design matrix Z
        """

        Z = np.ones(((self.ylen*self.lags +1),Y[0].shape[0]))
        return self.create_design_matrix(Z, self.data, Y.shape[0], self.lags)

    def _forecast_mean(self,h,t_params,Y,shock_type=None,shock_index=0,shock_value=None,shock_dir='positive',irf_intervals=False):
        """ Function allows for mean prediction; also allows shock specification for simulations or impulse response effects

        Parameters
        ----------
        h : int
            How many steps ahead to forecast

        t_params : np.array
            Transformed latent variables vector

        Y : np.array
            Data for series that is being forecast

        shock_type : None or str
            Type of shock; options include None, 'Cov' (simulate from covariance matrix), 'IRF' (impulse response shock)

        shock_index : int
            Which latent variable to apply the shock to if using an IRF.

        shock_value : None or float
            If specified, applies a custom-sized impulse response shock.

        shock_dir : str
            Direction of the IRF shock. One of 'positive' or 'negative'.

        irf_intervals : Boolean
            Whether to have intervals for the IRF plot or not

        Returns
        ----------
        A vector of forecasted data
        """         

        random = self._shock_create(h, shock_type, shock_index, shock_value, shock_dir,irf_intervals)
        exp = [Y[variable] for variable in range(0,self.ylen)]
        
        # Each forward projection
        for t in range(0,h):
            new_values = np.zeros(self.ylen)

            # Each variable
            for variable in range(0,self.ylen):
                index_ref = variable*(1+self.ylen*self.lags)
                new_values[variable] = t_params[index_ref] # constant

                # VAR(p) terms
                for lag in range(0,self.lags):
                    for lagged_var in range(0,self.ylen):
                        new_values[variable] += t_params[index_ref+lagged_var+(lag*self.ylen)+1]*exp[lagged_var][-1-lag]
                
                # Random shock
                new_values[variable] += random[t][variable]

            # Add new values
            for variable in range(0,self.ylen):
                exp[variable] = np.append(exp[variable],new_values[variable])

        return np.array(exp)

    def _model(self,beta):
        """ Creates the structure of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        mu : np.array
            Contains the predicted values for the time series

        Y : np.array
            Contains the length-adjusted time series (accounting for lags)
        """     

        Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])

        # Transform latent variables
        beta = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])

        params = []
        col_length = 1 + self.ylen*self.lags
        for i in range(0,self.ylen):
            params.append(beta[(col_length*i): (col_length*(i+1))])

        mu = np.dot(np.array(params),self._create_Z(Y))
        return mu, Y

    def _shock_create(self, h, shock_type, shock_index, shock_value, shock_dir, irf_intervals):
        """ Function creates shocks based on desired specification

        Parameters
        ----------
        h : int
            How many steps ahead to forecast

        shock_type : None or str
            Type of shock; options include None, 'Cov' (simulate from covariance matrix), 'IRF' (impulse response shock)

        shock_index : int
            Which latent variables to apply the shock to if using an IRF.

        shock_value : None or float
            If specified, applies a custom-sized impulse response shock.

        shock_dir : str
            Direction of the IRF shock. One of 'positive' or 'negative'.

        irf_intervals : Boolean
            Whether to have intervals for the IRF plot or not

        Returns
        ----------
        A h-length list which contains np.arrays containing shocks for each variable
        """     

        if shock_type is None:

            random = [np.zeros(self.ylen) for i in range(0,h)]

        elif shock_type == 'IRF':

            if self.use_ols_covariance is False:
                cov = self.custom_covariance(self.latent_variables.get_z_values())
            else:
                cov = self.ols_covariance()

            post = ss.multivariate_normal(np.zeros(self.ylen),cov)
            
            if irf_intervals is False:
                random = [np.zeros(self.ylen) for i in range(0,h)]
            else:
                random = [post.rvs() for i in range(0,h)]
                random[0] = np.zeros(self.ylen)

            if shock_value is None:
                if shock_dir=='positive':
                    random[0][shock_index] = cov[shock_index,shock_index]**0.5
                elif shock_dir=='negative':
                    random[0][shock_index] = -cov[shock_index,shock_index]**0.5
                else:
                    raise ValueError("Unknown shock direction!")    
            else:
                random[0][shock_index] = shock_value        

        elif shock_type == 'Cov':
            
            if self.use_ols_covariance is False:
                cov = self.custom_covariance(self.latent_variables.get_z_values())
            else:
                cov = self.ols_covariance()

            post = ss.multivariate_normal(np.zeros(self.ylen),cov)
            random = [post.rvs() for i in range(0,h)]

        return random

    def _summarize_simulations(self,mean_values,sim_vector,date_index,h,past_values):
        """ Summarizes a simulation vector and a mean vector of predictions

        Parameters
        ----------
        mean_values : np.array
            Mean predictions for h-step ahead forecasts

        sim_vector : np.array
            N simulation predictions for h-step ahead forecasts

        date_index : pd.DateIndex or np.array
            Dates for the simulations

        h : int
            How many steps ahead are forecast

        past_values : int
            How many past observations to include in the forecast plot

        intervals : Boolean
            Would you like to show prediction intervals for the forecast?
        """             

        error_bars = []
        for pre in range(5,100,5):
            error_bars.append(np.insert([np.percentile(i,pre) for i in sim_vector] - mean_values[-h:],0,0))
        forecasted_values = mean_values[-h-1:]
        plot_values = mean_values[-h-past_values:]
        plot_index = date_index[-h-past_values:]
        return error_bars, forecasted_values, plot_values, plot_index

    def custom_covariance(self,beta):
        """ Creates Covariance Matrix for a given Beta Vector
        (Not necessarily the OLS covariance)

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        A Covariance Matrix
        """         

        cov_matrix = np.zeros((self.ylen,self.ylen))
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        return custom_covariance_matrix(cov_matrix, self.ylen, self.lags, parm)

    def estimator_cov(self,method):
        """ Creates covariance matrix for the estimators

        Parameters
        ----------
        method : str
            Estimation method

        Returns
        ----------
        A Covariance Matrix
        """         
        
        Y = np.array([reg[self.lags:] for reg in self.data])    
        Z = self._create_Z(Y)
        if method == 'OLS':
            sigma = self.ols_covariance()
        else:           
            sigma = self.custom_covariance(self.latent_variables.get_z_values())
        return np.kron(np.linalg.inv(np.dot(Z,np.transpose(Z))), sigma)

    def neg_loglik(self,beta):
        """ Creates the negative log-likelihood of the model

        Parameters
        ----------
        beta : np.array
            Contains untransformed starting values for latent variables

        Returns
        ----------
        The negative logliklihood of the model
        """     

        mu, Y = self._model(beta)

        if self.use_ols_covariance is False:
            cm = self.custom_covariance(beta)
        else:
            cm = self.ols_covariance()

        diff = Y.T - mu.T
        ll1 =  -(mu.T.shape[0]*mu.T.shape[1]/2.0)*np.log(2.0*np.pi) - (mu.T.shape[0]/2.0)*np.linalg.slogdet(cm)[1]
        inverse = np.linalg.pinv(cm)

        return var_likelihood(ll1, mu.T.shape[0], diff, inverse)

    def ols_covariance(self):
        """ Creates OLS estimate of the covariance matrix

        Returns
        ----------
        The OLS estimate of the covariance matrix
        """         

        Y = np.array([reg[self.lags:reg.shape[0]] for reg in self.data])        
        return (1.0/(Y[0].shape[0]))*np.dot(self.residuals(Y),np.transpose(self.residuals(Y)))

    def plot_fit(self,**kwargs):
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
            date_index = self.index[self.lags:self.data[0].shape[0]]
            mu, Y = self._model(self.latent_variables.get_z_values())
            for series in range(0,Y.shape[0]):
                plt.figure(figsize=figsize)
                plt.plot(date_index,Y[series],label='Data ' + str(series))
                plt.plot(date_index,mu[series],label='Filter' + str(series),c='black')  
                plt.title(self.data_name[series])
                plt.legend(loc=2)   
            plt.show()              

    def plot_predict(self,h=5,past_values=20,intervals=True,**kwargs):

        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        past_values : int (default : 20)
            How many past observations to show on the forecast graph?

        intervals : Boolean
            Would you like to show prediction intervals for the forecast?

        Returns
        ----------
        - Plot of the forecast
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        if self.latent_variables.estimated is False:
            raise Exception("No latent varaibles estimated!")
        else:

            # Retrieve data, dates and (transformed) latent variables
            mu, Y = self._model(self.latent_variables.get_z_values()) 
            date_index = self.shift_dates(h)
            t_params = self.transform_z()

            # Expectation
            exps = self._forecast_mean(h,t_params,Y,None,None)

            # Simulation
            sim_vector = np.array([np.zeros([15000,h]) for i in range(self.ylen)])
            for it in range(0,15000):
                exps_sim = self._forecast_mean(h,t_params,Y,"Cov",None)
                for variable in range(self.ylen):
                    sim_vector[variable][it,:] = exps_sim[variable][-h:]

            for variable in range(0,exps.shape[0]):
                test = np.transpose(sim_vector[variable])
                error_bars, forecasted_values, plot_values, plot_index = self._summarize_simulations(exps[variable],test,date_index,h,past_values)
                plt.figure(figsize=figsize)
                if intervals == True:
                    alpha = [0.15*i/float(100) for i in range(50,12,-2)]
                    for count, pre in enumerate(error_bars):
                        plt.fill_between(date_index[-h-1:], forecasted_values-pre, forecasted_values+pre,alpha=alpha[count])            
                plt.plot(plot_index,plot_values)
                plt.title("Forecast for " + self.data_name[variable])
                plt.xlabel("Time")
                plt.ylabel(self.data_name[variable])
                plt.show()

    def predict_is(self, h=5, fit_once=False, fit_method='OLS', **kwargs):
        """ Makes dynamic in-sample predictions with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps would you like to forecast?

        fit_once : boolean
            Whether to fit the model once at the beginning (True), or fit every iteration (False)

        fit_method : string
            Which method to fit the model with

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        iterations = kwargs.get('iterations', 1000)

        predictions = []

        for t in range(0,h):
            new_data = self.data_original.iloc[:-h+t]
            x = VAR(lags=self.lags, integ=self.integ, data=new_data)
            
            if fit_once is False:
                if fit_method == 'BBVI':
                    x.fit(fit_method='BBVI', iterations=iterations)
                else:
                    x.fit(fit_method=fit_method)

            if t == 0:

                if fit_once is True:
                    if fit_method == 'BBVI':
                        x.fit(fit_method='BBVI', iterations=iterations)
                    else:
                        x.fit(fit_method=fit_method)
                    saved_lvs = x.latent_variables

                predictions = x.predict(1)
            else:
                if fit_once is True:
                    x.latent_variables = saved_lvs
                    
                predictions = pd.concat([predictions,x.predict(1)])
        
        #predictions.rename(columns={0:self.data_name}, inplace=True)
        predictions.index = self.index[-h:]

        return predictions

    def predict(self,h=5):
        """ Makes forecast with the estimated model

        Parameters
        ----------
        h : int (default : 5)
            How many steps ahead would you like to forecast?

        Returns
        ----------
        - pd.DataFrame with predicted values
        """     

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:

            # Retrieve data, dates and (transformed) latent variables
            mu, Y = self._model(self.latent_variables.get_z_values()) 
            date_index = self.shift_dates(h)
            t_params = self.transform_z()

            # Expectation
            exps = self._forecast_mean(h,t_params,Y,None,None)

            for variable in range(0,exps.shape[0]):
                forecasted_values = exps[variable][-h:]
                if variable == 0:
                    result = pd.DataFrame(forecasted_values)
                    result.rename(columns={0:self.data_name[variable]}, inplace=True)
                    result.index = date_index[-h:]          
                else:
                    result[self.data_name[variable]] = forecasted_values    

            return result

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

        date_index = self.index[-h:]
        predictions = self.predict_is(h)

        for variable in range(self.ylen):
            plt.figure(figsize=figsize)
            data = self.data[variable][-h:]
            plt.plot(date_index,data,label=self.data_name[variable] + ' Data')
            plt.plot(date_index,predictions.ix[:,variable].values,label='Predictions',c='black')
            plt.title(self.data_name[variable])
            plt.legend(loc=2)   
            plt.show()      

    def residuals(self,Y):
        """ Creates the model residuals

        Parameters
        ----------
        Y : np.array
            The dependent variables Y

        Returns
        ----------
        The model residuals
        """         

        return (Y-np.dot(self._create_B(Y),self._create_Z(Y)))

    def construct_wishart(self,v,X):
        """
        Constructs a Wishart prior for the covariance matrix
        """
        self.adjust_prior(list(range(int((len(self.latent_variables.z_list)-self.ylen-(self.ylen**2-self.ylen)/2)),
            int(len(self.latent_variables.z_list)))), fam.InverseWishart(v,X))
