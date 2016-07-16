import numpy as np

from .output import TablePrinter
from .tests import find_p_value
from .inference import norm_post_sim

class Results(object):

    def __init__(self):
        pass


class MLEResults(Results):

    def __init__(self,data_name,X_names,model_name,model_type,parameters, 
        results,data,index, multivariate_model,objective_object,method,
        param_hide,max_lag,ihessian=None,signal=None,scores=None,states=None,
        states_var=None):

        self.data_name = data_name
        self.X_names = X_names
        self.max_lag = max_lag
        self.model_name = model_name
        self.model_type = model_type
        self.parameters = parameters
        self.parameters_values = parameters.get_parameter_values()
        self.results = results
        self.method = method

        self.ihessian = ihessian
        self.scores = scores
        self.states = states
        self.states_var = states_var
        self.data = data
        self.index = index
        self.signal = signal
        self.multivariate_model = multivariate_model
        self.param_hide = int(param_hide)

        if self.multivariate_model is True:
            self.data_length = self.data[0].shape[0]
            self.data_name = ",".join(self.data_name)
        else:
            self.data_length = self.data.shape[0]

        self.objective_object = objective_object

        if self.method == 'MLE' or self.method == 'OLS':
            self.loglik = -self.objective_object(self.parameters_values)
            self.aic = 2*len(self.parameters_values)+2*self.objective_object(self.parameters_values)
            self.bic = 2*self.objective_object(self.parameters_values) + len(self.parameters_values)*np.log(self.data_length)
        elif self.method == 'PML':
            self.aic = 2*len(self.parameters_values)+2*self.objective_object(self.parameters_values)
            self.bic = 2*self.objective_object(self.parameters_values) + len(self.parameters_values)*np.log(self.data_length)

        if self.model_type in ['LLT','LLEV']:
            self.rounding_points = 10
        else:
            self.rounding_points = 4

    def __str__(self):
        if self.method == 'MLE':
            print("MLE Results Object")
        elif self.method == 'OLS':
            print("OLS Results Object")
        else:
            print("PML Results Object")
        print("==========================")
        print("Dependent variable: " + self.data_name)
        print("Regressors: " + str(self.X_names))
        print("==========================")
        print("Parameter Attributes: ")
        if self.ihessian is not None:
            print(".ihessian: Inverse Hessian")     
        print(".parameters : Parameters() object")
        if self.results is not None:
            print(".results : optimizer results")
        print("")
        print("Implied Model Attributes: ")
        print(".aic: Akaike Information Criterion") 
        print(".bic: Bayesian Information Criterion")   
        print(".data: Model Data")  
        print(".index: Model Index")
        if self.method == 'MLE' or self.method == 'OLS':
            print(".loglik: Loglikelihood") 
        if self.scores is not None:
            print(".scores: Model Scores")                      
        if self.signal is not None:
            print(".signal: Model Signal")          
        if self.states is not None:
            print(".states: Model States")      
        if self.states_var is not None:
            print(".states_var: Model State Variances")                                     
        print(".results : optimizer results")
        print("")       
        print("Methods: ")
        print(".summary() : printed results")
        return("")

    def summary(self):
        if self.ihessian is not None:
            return self.summary_with_hessian()
        else:
            return self.summary_without_hessian()

    def summary_with_hessian(self):

        ses = np.power(np.abs(np.diag(self.ihessian)),0.5)
        t_params = self.parameters.get_parameter_values(transformed=True)
        t_p_std = ses.copy() # vector for transformed standard errors

        # Create transformed variables
        for k in range(len(t_params)-self.param_hide):
            z_temp = (self.parameters_values[k]/float(ses[k]))
            t_p_std[k] = t_params[k] / z_temp

        data = []

        for i in range(len(self.parameters.parameter_list)-self.param_hide):
            if self.parameters.parameter_list[i].prior.transform == np.array:
                data.append({
                    'param_name': self.parameters.parameter_list[i].name, 
                    'param_value':np.round(self.parameters.parameter_list[i].prior.transform(self.parameters_values[i]),self.rounding_points), 
                    'param_std': np.round(t_p_std[i],self.rounding_points),
                    'param_z': np.round(t_params[i]/float(t_p_std[i]),self.rounding_points),
                    'param_p': np.round(find_p_value(t_params[i]/float(t_p_std[i])),self.rounding_points),
                    'ci': "(" + str(np.round(t_params[i] - t_p_std[i]*1.96,self.rounding_points)) + " | " + str(np.round(t_params[i] + t_p_std[i]*1.96,self.rounding_points)) + ")"})
            else:
                data.append({
                    'param_name': self.parameters.parameter_list[i].name, 
                    'param_value':np.round(self.parameters.parameter_list[i].prior.transform(self.parameters_values[i]),self.rounding_points)})                     
        
        fmt = [
            ('Parameter',       'param_name',   40),
            ('Estimate',          'param_value',       10),
            ('Std Error', 'param_std', 10),
            ('z',          'param_z',       8),
            ('P>|z|',          'param_p',       8),
            ('95% C.I.',          'ci',       25)
            ]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        if self.method == 'MLE' or self.method == 'OLS' :
            obj_desc = "Log Likelihood: " + str(np.round(-self.objective_object(self.parameters_values),4))
        else:
            obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective_object(self.parameters_values),4))

        model_details.append({'model_details': 'Dependent Variable: ' + str(self.data_name), 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(np.round(2*len(self.parameters_values)+2*self.objective_object(self.parameters_values),4))})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(np.round(2*self.objective_object(self.parameters_values) + len(self.parameters_values)*np.log(self.data_length),4))})


        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)

    def summary_without_hessian(self):

        t_params = self.parameters.get_parameter_values(transformed=True)

        print ("Hessian not invertible! Consider a different model specification.")
        print ("")      

        data = []

        for i in range(len(self.parameters.parameter_list)):
            data.append({'param_name': self.parameters.parameter_list[i].name, 'param_value':np.round(self.parameters.parameter_list[i].prior.transform(self.results.x[i]),4)})

        fmt = [
            ('Parameter',       'param_name',   40),
            ('Estimate',          'param_value',       10)
            ]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        if self.method == 'MLE':
            obj_desc = "Log Likelihood: " + str(np.round(-self.objective_object(self.results.x),4))
        else:
            obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective_object(self.results.x),4))

        model_details.append({'model_details': 'Dependent Variable: ' + self.data_name, 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(self.aic)})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(self.bic)})


        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)


class BBVIResults(Results):

    def __init__(self,data_name,X_names,model_name,model_type,parameters, 
        data,index,multivariate_model,objective_object,method,
        param_hide,max_lag,ses,signal=None,scores=None,states=None,
        states_var=None):

        self.data_name = data_name
        self.X_names = X_names
        self.max_lag = max_lag
        self.model_name = model_name
        self.model_type = model_type
        self.parameters = parameters
        self.method = method

        self.ses = ses
        self.scores = scores
        self.states = states
        self.states_var = states_var
        self.data = data
        self.index = index
        self.signal = signal
        self.multivariate_model = multivariate_model
        self.param_hide = param_hide

        if self.multivariate_model is True:
            self.data_length = self.data[0].shape[0]
            self.data_name = ",".join(self.data_name)
        else:
            self.data_length = self.data.shape[0]

        params = self.parameters.get_parameter_values(transformed=False)

        self.objective_object = objective_object
        self.aic = 2*len(params)+2*self.objective_object(params)
        self.bic = 2*self.objective_object(params) + len(params)*np.log(self.data_length)

        if self.model_type in ['LLT','LLEV']:
            self.rounding_points = 10
        else:
            self.rounding_points = 4

    def __str__(self):
        print("BBVI Results Object")
        print("==========================")
        print("Dependent variable: " + self.data_name)
        print("Regressors: " + str(self.X_names))
        print("==========================")
        print("Parameter Attributes: ")
        print(".parameters : Parameters() object")
        print(".results : optimizer results")
        print("")
        print("Implied Model Attributes: ")
        print(".aic: Akaike Information Criterion") 
        print(".bic: Bayesian Information Criterion")   
        print(".data: Model Data")  
        print(".index: Model Index")
        if self.scores is not None:
            print(".scores: Model Scores")                      
        if self.signal is not None:
            print(".signal: Model Signal")          
        if self.states is not None:
            print(".states: Model States")  
        if self.states_var is not None:
            print(".states_var: Model State Variances")                                     
        print("")       
        print("Methods: ")
        print(".summary() : printed results")
        return("")

    def summary(self):
        ihessian = np.diag(np.power(np.exp(self.ses),2))
        params = self.parameters.get_parameter_values(transformed=False)
        chain, mean_est, median_est, upper_95_est, lower_95_est = norm_post_sim(params,ihessian)

        for k in range(len(chain)):
            chain[k] = self.parameters.parameter_list[k].prior.transform(chain[k])
            mean_est[k] = self.parameters.parameter_list[k].prior.transform(mean_est[k])
            median_est[k] = self.parameters.parameter_list[k].prior.transform(median_est[k])
            upper_95_est[k] = self.parameters.parameter_list[k].prior.transform(upper_95_est[k])
            lower_95_est[k] = self.parameters.parameter_list[k].prior.transform(lower_95_est[k])                

        mean_est = np.array(mean_est)
        self.chains = chain[:]

        data = []

        for i in range(len(self.parameters.parameter_list)-self.param_hide):
            data.append({
                'param_name': self.parameters.parameter_list[i].name, 
                'param_mean': np.round(mean_est[i],self.rounding_points),
                'param_median': np.round(median_est[i],self.rounding_points),
                'ci':  "(" + str(np.round(lower_95_est[i],self.rounding_points)) + " | " + str(np.round(upper_95_est[i],self.rounding_points)) + ")"
                })                  

        fmt = [
            ('Parameter','param_name',40),
            ('Median','param_median',18),
            ('Mean', 'param_mean', 18),
            ('95% Credibility Interval','ci',25)]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        if self.method == 'MLE':
            obj_desc = "Log Likelihood: " + str(np.round(-self.objective_object(params),4))
        else:
            obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective_object(params),4))

        model_details.append({'model_details': 'Dependent Variable: ' + self.data_name, 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(self.aic)})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(self.bic)})

        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)


class BBVISSResults(Results):

    def __init__(self,data_name,X_names,model_name,model_type,parameters, 
        data,index,multivariate_model,objective,method,
        param_hide,max_lag,ses,signal=None,scores=None,states=None,
        states_var=None):

        self.data_name = data_name
        self.X_names = X_names
        self.max_lag = max_lag
        self.model_name = model_name
        self.model_type = model_type
        self.parameters = parameters
        self.method = method

        self.ses = ses
        self.scores = scores
        self.states = states
        self.states_var = states_var
        self.data = data
        self.index = index
        self.signal = signal
        self.multivariate_model = multivariate_model
        self.param_hide = param_hide

        if self.multivariate_model is True:
            self.data_length = self.data[0].shape[0]
            self.data_name = ",".join(self.data_name)
        else:
            self.data_length = self.data.shape[0]

        params = self.parameters.get_parameter_values(transformed=False)

        self.objective = objective
        self.aic = 2*len(params)+2*self.objective
        self.bic = 2*self.objective + len(params)*np.log(self.data_length)

        if self.model_type in ['LLT','LLEV']:
            self.rounding_points = 10
        else:
            self.rounding_points = 4

    def __str__(self):
        print("BBVI Results Object")
        print("==========================")
        print("Dependent variable: " + self.data_name)
        print("Regressors: " + str(self.X_names))
        print("==========================")
        print("Parameter Attributes: ")
        print(".parameters : Parameters() object")
        print("")
        print("Implied Model Attributes: ")
        print(".aic: Akaike Information Criterion") 
        print(".bic: Bayesian Information Criterion")   
        print(".data: Model Data")  
        print(".index: Model Index")
        print(".objective: Unnormalized Log Posterior")
        if self.scores is not None:
            print(".scores: Model Scores")                      
        if self.signal is not None:
            print(".signal: Model Signal")          
        if self.states is not None:
            print(".states: Model States")  
        if self.states_var is not None:
            print(".states_var: Model State Variances")                                     
        print("")       
        print("Methods: ")
        print(".summary() : printed results")
        return("")

    def summary(self):
        ihessian = np.diag(np.power(np.exp(self.ses),2))
        params = self.parameters.get_parameter_values(transformed=False)
        chain, mean_est, median_est, upper_95_est, lower_95_est = norm_post_sim(params,ihessian)
        for k in range(len(params)):
            chain[k] = self.parameters.parameter_list[k].prior.transform(chain[k])
            mean_est[k] = self.parameters.parameter_list[k].prior.transform(mean_est[k])
            median_est[k] = self.parameters.parameter_list[k].prior.transform(median_est[k])
            upper_95_est[k] = self.parameters.parameter_list[k].prior.transform(upper_95_est[k])
            lower_95_est[k] = self.parameters.parameter_list[k].prior.transform(lower_95_est[k])                

        mean_est = np.array(mean_est)
        self.chains = chain[:]

        data = []

        for i in range(len(self.parameters.parameter_list)-self.param_hide):
            data.append({
                'param_name': self.parameters.parameter_list[i].name, 
                'param_mean': np.round(mean_est[i],self.rounding_points),
                'param_median': np.round(median_est[i],self.rounding_points),
                'ci':  "(" + str(np.round(lower_95_est[i],self.rounding_points)) + " | " + str(np.round(upper_95_est[i],self.rounding_points)) + ")"
                })                  

        fmt = [
            ('Parameter','param_name',40),
            ('Median','param_median',18),
            ('Mean', 'param_mean', 18),
            ('95% Credibility Interval','ci',25)]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective,4))

        model_details.append({'model_details': 'Dependent Variable: ' + self.data_name, 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(self.aic)})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(self.bic)})

        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)


class LaplaceResults(Results):

    def __init__(self,data_name,X_names,model_name,model_type,parameters, 
        data,index,multivariate_model,objective_object,method,
        param_hide,max_lag,ihessian,signal=None,scores=None,states=None,
        states_var=None):

        self.data_name = data_name
        self.X_names = X_names
        self.max_lag = max_lag
        self.model_name = model_name
        self.model_type = model_type
        self.parameters = parameters
        self.method = method

        self.ihessian = ihessian
        self.scores = scores
        self.states = states
        self.states_var = states_var
        self.data = data
        self.index = index
        self.signal = signal
        self.multivariate_model = multivariate_model
        self.param_hide = param_hide

        if self.multivariate_model is True:
            self.data_length = self.data[0].shape[0]
            self.data_name = ",".join(self.data_name)
        else:
            self.data_length = self.data.shape[0]

        params = self.parameters.get_parameter_values(transformed=False)

        self.objective_object = objective_object
        self.aic = 2*len(params)+2*self.objective_object(params)
        self.bic = 2*self.objective_object(params) + len(params)*np.log(self.data_length)

        if self.model_type in ['LLT','LLEV']:
            self.rounding_points = 10
        else:
            self.rounding_points = 4

    def __str__(self):
        print("Laplace Results Object")
        print("==========================")
        print("Dependent variable: " + self.data_name)
        print("Regressors: " + str(self.X_names))
        print("==========================")
        print("Parameter Attributes: ")
        if self.ihessian is not None:
            print(".ihessian: Inverse Hessian")             
        print(".parameters : Parameters() object")
        print(".results : optimizer results")
        print("")
        print("Implied Model Attributes: ")
        print(".aic: Akaike Information Criterion") 
        print(".bic: Bayesian Information Criterion")   
        print(".data: Model Data")  
        print(".index: Model Index")
        if self.scores is not None:
            print(".scores: Model Scores")                      
        if self.signal is not None:
            print(".signal: Model Signal")          
        if self.states is not None:
            print(".states: Model States")  
        if self.states_var is not None:
            print(".states_var: Model State Variances")                                     
        print("")       
        print("Methods: ")
        print(".summary() : printed results")
        return("")

    def summary(self):
        params = self.parameters.get_parameter_values(transformed=False)
        chain, mean_est, median_est, upper_95_est, lower_95_est = norm_post_sim(params,self.ihessian)

        for k in range(len(chain)):
            chain[k] = self.parameters.parameter_list[k].prior.transform(chain[k])
            mean_est[k] = self.parameters.parameter_list[k].prior.transform(mean_est[k])
            median_est[k] = self.parameters.parameter_list[k].prior.transform(median_est[k])
            upper_95_est[k] = self.parameters.parameter_list[k].prior.transform(upper_95_est[k])
            lower_95_est[k] = self.parameters.parameter_list[k].prior.transform(lower_95_est[k])                

        mean_est = np.array(mean_est)
        self.chains = chain[:]

        data = []

        for i in range(len(self.parameters.parameter_list)-self.param_hide):
            data.append({
                'param_name': self.parameters.parameter_list[i].name, 
                'param_mean': np.round(mean_est[i],self.rounding_points),
                'param_median': np.round(median_est[i],self.rounding_points),
                'ci':  "(" + str(np.round(lower_95_est[i],self.rounding_points)) + " | " + str(np.round(upper_95_est[i],self.rounding_points)) + ")"
                })                  

        fmt = [
            ('Parameter','param_name',40),
            ('Median','param_median',18),
            ('Mean', 'param_mean', 18),
            ('95% Credibility Interval','ci',25)]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        if self.method == 'MLE':
            obj_desc = "Log Likelihood: " + str(np.round(-self.objective_object(params),4))
        else:
            obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective_object(params),4))

        model_details.append({'model_details': 'Dependent Variable: ' + self.data_name, 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(self.aic)})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(self.bic)})

        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)

class MCMCResults(Results):

    def __init__(self,data_name,X_names,model_name,model_type,parameters, 
        data,index,multivariate_model,objective_object,method,
        param_hide,max_lag,samples,mean_est,median_est,lower_95_est,upper_95_est,
        signal=None,scores=None,states=None,states_var=None):
        self.data_name = data_name
        self.X_names = X_names
        self.max_lag = max_lag
        self.model_name = model_name
        self.model_type = model_type
        self.parameters = parameters
        self.method = method

        self.samples = samples
        self.mean_est = mean_est
        self.median_est = median_est
        self.lower_95_est = lower_95_est
        self.upper_95_est = upper_95_est
        self.scores = scores
        self.states = states
        self.states_var = states_var
        self.data = data
        self.index = index
        self.signal = signal
        self.multivariate_model = multivariate_model
        self.param_hide = param_hide

        if self.multivariate_model is True:
            self.data_length = self.data[0].shape[0]
            self.data_name = ",".join(self.data_name)
        else:
            self.data_length = self.data.shape[0]

        params = self.parameters.get_parameter_values(transformed=False)

        self.objective_object = objective_object
        self.aic = 2*len(params)+2*self.objective_object(params)
        self.bic = 2*self.objective_object(params) + len(params)*np.log(self.data_length)

        if self.model_type in ['LLT','LLEV']:
            self.rounding_points = 10
        else:
            self.rounding_points = 4

    def __str__(self):
        print("Metropolis Hastings Results Object")
        print("==========================")
        print("Dependent variable: " + self.data_name)
        print("Regressors: " + str(self.X_names))
        print("==========================")
        print("Parameter Attributes: ")     
        print(".parameters : Parameters() object")
        print(".results : optimizer results")
        if self.samples is not None:
            print(".samples: MCMC samples")             
        print("")
        print("Implied Model Attributes: ")
        print(".aic: Akaike Information Criterion") 
        print(".bic: Bayesian Information Criterion")   
        print(".data: Model Data")  
        print(".index: Model Index")
        if self.scores is not None:
            print(".scores: Model Scores")                      
        if self.signal is not None:
            print(".signal: Model Signal")          
        if self.states is not None:
            print(".states: Model States")  
        if self.states_var is not None:
            print(".states_var: Model State Variances")                                     
        print("")       
        print("Methods: ")
        print(".summary() : printed results")
        return("")

    def summary(self):
        params = self.parameters.get_parameter_values(transformed=False)

        data = []

        for i in range(len(self.parameters.parameter_list)-self.param_hide):
            data.append({
                'param_name': self.parameters.parameter_list[i].name, 
                'param_mean': np.round(self.mean_est[i],self.rounding_points),
                'param_median': np.round(self.median_est[i],self.rounding_points),
                'ci':  "(" + str(np.round(self.lower_95_est[i],self.rounding_points)) + " | " + str(np.round(self.upper_95_est[i],self.rounding_points)) + ")"
                })                  

        fmt = [
            ('Parameter','param_name',40),
            ('Median','param_median',18),
            ('Mean', 'param_mean', 18),
            ('95% Credibility Interval','ci',25)]

        model_details = []

        model_fmt = [
            (self.model_name, 'model_details', 55),
            ('', 'model_results', 50)
            ]

        if self.method == 'MLE':
            obj_desc = "Log Likelihood: " + str(np.round(-self.objective_object(params),4))
        else:
            obj_desc = "Unnormalized Log Posterior: " + str(np.round(-self.objective_object(params),4))

        model_details.append({'model_details': 'Dependent Variable: ' + self.data_name, 
            'model_results': 'Method: ' + str(self.method)})
        model_details.append({'model_details': 'Start Date: ' + str(self.index[self.max_lag]),
            'model_results': obj_desc})
        model_details.append({'model_details': 'End Date: ' + str(self.index[-1]),
            'model_results': 'AIC: ' + str(self.aic)})
        model_details.append({'model_details': 'Number of observations: ' + str(self.data_length),
            'model_results': 'BIC: ' + str(self.bic)})

        print( TablePrinter(model_fmt, ul='=')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)
