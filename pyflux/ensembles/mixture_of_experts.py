import pandas as pd
import numpy as np

class Aggregate():
    """ Aggregation Algorithm

    Parameters
    ----------
    learning_rate : float
        Learning rate for the Aggregation algorithm

    loss_type : string
        'absolute' or 'squared'

    match_window : int
        (default: 10) how many of the observations at the end of a model time 
        series to assess whether the data being used in each model is the same
    """

    def __init__(self, learning_rate=1.0, loss_type='absolute', match_window=10):
        self.learning_rate = learning_rate
        self.data = []
        self.model_list = []
        self.model_names = []
        self.match_window = match_window
        self.model_predictions_is = []
        self.model_predictions = []

        if loss_type == 'absolute':
            self.loss_type = self.absolute_loss
            self.loss_name = 'Absolute Loss'
        elif loss_type == 'squared':
            self.loss_type = self.squared_loss
            self.loss_name = 'Squared Loss'
        else:
            raise ValueError('Unidentified loss type entered!')

        self.supported_models = ['EGARCH', 'EGARCHM', 'EGARCHMReg', 'GARCH', 'LMEGARCH', 'LMSEGARCH', 'SEGARCH', 'SEGARCHM', 'GAS', 'ARIMA', 'ARIMAX', 'GASLLEV', 'GASLLT', 'GASReg', 'GASX', 'GPNARX', 'LLEV', 'LLT', 'DynReg']

    @staticmethod
    def absolute_loss(data, predictions):
        """ Calculates absolute loss 
        
        Parameters
        ----------
        data : np.ndarray
            Univariate data

        predictions : np.ndarray
            Univariate predictions

        Returns
        ----------
        - np.ndarray of the absolute loss
        """
        return np.abs(data-predictions)     

    @staticmethod
    def squared_loss(data, predictions):
        """ Calculates squared loss
        
        Parameters
        ----------
        data : np.ndarray
            Univariate data

        predictions : np.ndarray
            Univariate predictions

        Returns
        ----------
        - np.ndarray of the squared loss
        """
        return np.square(data-predictions)

    def add_model(self, model):
        """ Adds a PyFlux model to the aggregating algorithm
        
        Parameters
        ----------
        model : pf.[MODEL]
            A PyFlux univariate model

        Returns
        ----------
        - Void (changes self.model_list)
        """

        if model.model_type not in self.supported_models:
            raise ValueError('Model type not supported for Aggregate! Apologies')

        if not self.model_list:
            self.model_list.append(model)
            if model.model_type in ['EGARCH', 'EGARCHM', 'EGARCHMReg', 'GARCH', 'LMEGARCH', 'LMSEGARCH', 'SEGARCH', 'SEGARCHM']:
                self.data = np.abs(model.data)
            else:
                self.data = model.data
            self.index = model.index
        else:
            if model.model_type in ['EGARCH', 'EGARCHM', 'EGARCHMReg', 'GARCH', 'LMEGARCH', 'LMSEGARCH', 'SEGARCH', 'SEGARCHM']:
                if np.isclose(np.abs(np.abs(model.data[-self.match_window:])-self.data[-self.match_window:]).sum(),0.0) or model.model_type=='GPNARX':
                    self.model_list.append(model)
                else:
                    raise ValueError('Data entered is deemed different based on %s last values!' % (s))
            else:
                if np.isclose(np.abs(model.data[-self.match_window:]-self.data[-self.match_window:]).sum(),0.0) or model.model_type=='GPNARX':
                    self.model_list.append(model)
                else:
                    raise ValueError('Data entered is deemed different based on %s last values!' % (s))

        self.model_names = [i.model_name for i in self.model_list]

    def _model_predict(self, h, recalculate=False, fit_once=True):
        """ Outputs ensemble model predictions for out-of-sample data
        
        Parameters
        ----------
        h : int
            How many steps at the end of the series to run the ensemble on

        recalculate: boolean
            Whether to recalculate the predictions or not

        fit_once : boolean
            Whether to fit the model once at the beginning, or with every iteration
        
        Returns
        ----------
        - pd.DataFrame of the model predictions, index of dates
        """

        if len(self.model_predictions) == 0 or h != self.h or recalculate is True:

            for no, model in enumerate(self.model_list):
                if no == 0:
                    model.fit()
                    result = model.predict(h)
                    self.predict_index = result.index
                    result.columns = [model.model_name]
                else:
                    model.fit()
                    new_frame = model.predict(h)
                    new_frame.columns = [model.model_name]
                    result = pd.concat([result,new_frame], axis=1)

            self.model_predictions = result
            self.h = h
            return result, self.predict_index

        else:
            return self.model_predictions, self.predict_index

    def _model_predict_is(self, h, recalculate=False, fit_once=True):
        """ Outputs ensemble model predictions for the end-of-period data
        
        Parameters
        ----------
        h : int
            How many steps at the end of the series to run the ensemble on

        recalculate: boolean
            Whether to recalculate the predictions or not

        fit_once : boolean
            Whether to fit the model once at the beginning, or with every iteration
        
        Returns
        ----------
        - pd.DataFrame of the model predictions, index of dates
        """

        if len(self.model_predictions_is) == 0 or h != self.h or recalculate is True:

            for no, model in enumerate(self.model_list):
                if no == 0:
                    result = model.predict_is(h, fit_once=fit_once)
                    result.columns = [model.model_name]
                else:
                    new_frame = model.predict_is(h, fit_once=fit_once)
                    new_frame.columns = [model.model_name]
                    result = pd.concat([result,new_frame], axis=1)

            self.model_predictions_is = result
            self.h = h
            return result

        else:
            return self.model_predictions_is

    def _construct_losses(self, data, predictions, ensemble_prediction):
        """ Construct losses for the ensemble and each constitute model
        
        Parameters
        ----------
        data: np.ndarray
            The univariate time series

        predictions : np.ndarray
            The predictions of each constitute model

        ensemble_prediction : np.ndarray
            The prediction of the ensemble model

        Returns
        ----------
        - np.ndarray of the losses for each model
        """

        losses = []
        losses.append(self.loss_type(data, ensemble_prediction).sum()/data.shape[0])
        for model in range(len(self.model_list)):
            losses.append(self.loss_type(data, predictions[:,model]).sum()/data.shape[0])
        return losses

    def tune_learning_rate(self, h, parameter_list=None):
        """ Naive tuning of the the learning rate on the in-sample data
        
        Parameters
        ----------
        h : int
            How many steps to run Aggregate on

        parameter_list: list
            List of parameters to search for a good learning rate over

        Returns
        ----------
        - Void (changes self.learning_rate)
        """

        if parameter_list is None:
            parameter_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0,100000.0]
        
        for parameter in parameter_list:
            self.learning_rate = parameter
            _, losses, _ = self.run(h, recalculate=False)
            loss = losses[0]
            if parameter == parameter_list[0]:
                best_rate = parameter
                best_loss = loss
            else:
                if loss < best_loss:
                    best_loss = loss
                    best_rate = parameter

        self.learning_rate = best_rate

    def run(self, h, recalculate=False):
        """ Run the aggregating algorithm 
        
        Parameters
        ----------
        h : int
            How many steps to run the aggregating algorithm on 

        recalculate: boolean
            Whether to recalculate the predictions or not

        Returns
        ----------
        - np.ndarray of normalized weights, np.ndarray of losses for each model
        """

        data = self.data[-h:] 
        predictions = self._model_predict_is(h, recalculate=recalculate).values
        weights = np.zeros((h, len(self.model_list)))
        normalized_weights = np.zeros((h, len(self.model_list)))
        ensemble_prediction = np.zeros(h)
        
        for t in range(h):
            if t == 0:
                weights[t,:] = 100000
                ensemble_prediction[t] = np.dot(weights[t,:]/weights[t,:].sum(), predictions[t,:])
                weights[t,:] = weights[t,:]*np.exp(-self.learning_rate*self.loss_type(data[t], predictions[t,:]))
                normalized_weights[t,:] = weights[t,:]/weights[t,:].sum()
            else:
                ensemble_prediction[t] = np.dot(weights[t-1,:]/weights[t-1,:].sum(), predictions[t,:])
                weights[t,:] = weights[t-1,:]*np.exp(-self.learning_rate*self.loss_type(data[t], predictions[t,:]))
                normalized_weights[t,:] = weights[t,:]/weights[t,:].sum()

        return normalized_weights, self._construct_losses(data, predictions, ensemble_prediction), ensemble_prediction

    def plot_weights(self, h, **kwargs):
        """ Plot the weights from the aggregating algorithm
        
        Parameters
        ----------
        h : int
            How many steps to run the aggregating algorithm on 

        Returns
        ----------
        - A plot of the weights for each model constituent over time
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(10,7))

        weights, _, _ = self.run(h=h)
        plt.figure(figsize=figsize)
        plt.plot(self.index[-h:],weights)
        plt.legend(self.model_names)
        plt.show()

    def predict(self, h, h_train=40):
        """ Run out-of-sample predicitons for Aggregate algorithm
        (This only works for non-exogenous variable models currently)
        
        Parameters
        ----------
        h : int
            How many out-of-sample steps to run the aggregating algorithm on 

        h_train : int
            How many in-sample steps to warm-up the ensemble weights on

        Returns
        ----------
        - pd.DataFrame of Aggregate out-of-sample predictions
        """

        predictions, index = self._model_predict(h)
        normalized_weights = self.run(h=h_train)[0][-1, :]
        ensemble_prediction = np.zeros(h)
        
        for t in range(h):
            ensemble_prediction[t] = np.dot(normalized_weights, predictions.values[t,:])

        result = pd.DataFrame(ensemble_prediction)
        result.index = index

        return result

    def predict_is(self, h):
        """ Outputs predictions for the Aggregate algorithm on the in-sample data

        Parameters
        ----------
        h : int
            How many steps to run the aggregating algorithm on 

        Returns
        ----------
        - pd.DataFrame of ensemble predictions
        """
        result = pd.DataFrame([self.run(h=h)[2]]).T  
        result.index = self.index[-h:]
        return result 

    def summary(self, h):
        """ 
        Summarize the results for each model for h steps of the algorithm

       Parameters
        ----------
        h : int
            How many steps to run the aggregating algorithm on 

        Returns
        ----------
        - pd.DataFrame of losses for each model       
        """
        _, losses, _ = self.run(h=h)
        df = pd.DataFrame(losses)
        df.index = ['Ensemble'] + self.model_names
        df.columns = [self.loss_name]
        return df
