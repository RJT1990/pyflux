import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as sp

from .. import families as fam
from .. import tsm as tsm
from .. import data_check as dc

from .scores import *

from .gas_core_recursions import gas_recursion

class GASRank(tsm.TSM):
    """ Inherits time series methods from TSM class.

    **** GENERALIZED AUTOREGRESSIVE SCORE (GAS) RANK MODELS ****

    Parameters
    ----------
    data : pd.DataFrame
        Field to specify the univariate time series data that will be used.

    team_1 : str (pd.DataFrame)        
        Specifies which column contains the home team

    team_2 : str (pd.DataFrame)        
        Specifies which column contains the away team

    family : GAS family object
        Which distribution to use, e.g. GASNormal()

    score_diff : str (pd.DataFrame)        
        Specifies which column contains the score

    gradient_only : Boolean (default: True)
        If true, will only use gradient rather than second-order terms
        to construct the modified score.
    """

    def __init__(self, data, team_1, team_2, family, score_diff, gradient_only=False):

        # Initialize TSM object     
        super(GASRank,self).__init__('GASRank')

        self.gradient_only = gradient_only
        self.z_no = 2
        self.max_lag = 0
        self._z_hide = 0 # Whether to cutoff variance latent variables from results
        self.supported_methods = ["MLE","PML","Laplace","M-H","BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        self.home_id, self.away_id = self._create_ids(data[team_1].values,data[team_2].values)
        self.team_strings = sorted(list(set(np.append(data[team_1].values,data[team_2].values))))
        self.team_dict = dict(zip(self.team_strings, range(len(self.team_strings))))
        self.home_count, self.away_count = self._match_count()
        self.max_team = max(np.max(self.home_id),np.max(self.away_id))
        self.original_dataframe = data
        self.data, self.data_name, self.is_pandas, self.index = dc.data_check(data, score_diff)
        self.data = self.data.astype(np.float) 
        self.data_original = self.data.copy()
        self.data_length = self.data.shape[0]

        self._create_latent_variables()

        self.family = family
        
        self.model_name2, self.link, self.scale, self.shape, self.skewness, self.mean_transform, _ = self.family.setup()
        self.model_name = self.model_name2 + "GAS Rank "

        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0],i[1],i[2])
            self.latent_variables.z_list[2+no].start = i[3]
        self.latent_variables.z_list[0].start = self.mean_transform(np.mean(self.data))

        self._model = self._model_one_components
        self._model_abilities = self._model_abilities_one_components
        self.plot_abilities = self.plot_abilities_one_components
        self.predict = self.predict_one_component

        self.family_z_no = len(self.family.build_latent_variables())
        self.z_no = len(self.latent_variables.z_list)

    def _create_ids(self, home_teams, away_teams):
        """
        Creates IDs for both players/teams
        """
        categories = pd.Categorical(np.append(home_teams,away_teams))
        home_id, away_id = categories.codes[0:int(len(categories)/2)], categories.codes[int(len(categories)/2):len(categories)+1]
        return home_id, away_id

    def _match_count(self):
        home_count, away_count = np.zeros(len(self.home_id)), np.zeros(len(self.away_id))

        for t in range(0,len(home_count)):
            home_count[t] = len(self.home_id[0:t+1][self.home_id[0:t+1]==self.home_id[t]]) + len(self.away_id[0:t+1][self.away_id[0:t+1]==self.home_id[t]]) 
            away_count[t] = len(self.home_id[0:t+1][self.home_id[0:t+1]==self.away_id[t]]) + len(self.away_id[0:t+1][self.away_id[0:t+1]==self.away_id[t]]) 

        return home_count, away_count       

    def _match_count_2(self):
        home_count, away_count = np.zeros(len(self.home_2_id)), np.zeros(len(self.away_2_id))

        for t in range(0,len(home_count)):
            home_count[t] = len(self.home_2_id[0:t+1][self.home_2_id[0:t+1]==self.home_2_id[t]]) + len(self.away_2_id[0:t+1][self.away_2_id[0:t+1]==self.home_2_id[t]]) 
            away_count[t] = len(self.home_2_id[0:t+1][self.home_2_id[0:t+1]==self.away_2_id[t]]) + len(self.away_2_id[0:t+1][self.away_2_id[0:t+1]==self.away_2_id[t]]) 

        return home_count, away_count       

    def _create_latent_variables(self):
        """ Creates model latent variables

        Returns
        ----------
        None (changes model attributes)
        """

        self.latent_variables.add_z('Constant', fam.Normal(0,10,transform=None), fam.Normal(0,3))
        self.latent_variables.add_z('Ability Scale', fam.Normal(0,1,transform=None), fam.Normal(0,3))

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

    def _model_one_components(self,beta):
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
        scale, shape, skewness = self._get_scale_and_shape(parm)
        state_vectors = np.zeros(shape=(self.max_team+1))
        theta = np.zeros(shape=(self.data.shape[0]))

        for t in range(0,self.data.shape[0]):
            theta[t] = parm[0] + state_vectors[self.home_id[t]] - state_vectors[self.away_id[t]]

            state_vectors[self.home_id[t]] += parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors[self.away_id[t]] += -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)

        return theta, self.data, state_vectors

    def _model_two_components(self,beta):
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
        scale, shape, skewness = self._get_scale_and_shape(parm)
        state_vectors_1 = np.zeros(shape=(self.max_team+1))
        state_vectors_2 = np.zeros(shape=(self.max_team_2+1))
        theta = np.zeros(shape=(self.data.shape[0]))

        for t in range(0,self.data.shape[0]):
            theta[t] = parm[0] + state_vectors_2[self.home_2_id[t]] - state_vectors_2[self.away_2_id[t]] + state_vectors_1[self.home_id[t]] - state_vectors_1[self.away_id[t]]

            state_vectors_1[self.home_id[t]] += parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_1[self.away_id[t]] += -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_2[self.home_2_id[t]] += parm[2]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_2[self.away_2_id[t]] += -parm[2]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)

        return theta, self.data, state_vectors_1

    def _model_abilities_one_components(self,beta):
        """ Creates the structure of the model - store abilities

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
        scale, shape, skewness = self._get_scale_and_shape(parm)
        state_vectors = np.zeros(shape=(self.max_team+1))
        state_vectors_store = np.zeros(shape=(int(np.max(self.home_count)+50),int(self.max_team+1)))
        theta = np.zeros(shape=(self.data.shape[0]))

        for t in range(0,self.data.shape[0]):
            theta[t] = parm[0] + state_vectors[self.home_id[t]] - state_vectors[self.away_id[t]]

            state_vectors[self.home_id[t]] += parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors[self.away_id[t]] += -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_store[int(self.home_count[t]), self.home_id[t]] = state_vectors_store[max(0,int(self.home_count[t])-1), self.home_id[t]] + parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_store[int(self.away_count[t]), self.away_id[t]] = state_vectors_store[max(0,int(self.away_count[t])-1), self.away_id[t]] -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)

        return state_vectors_store

    def _model_abilities_two_components(self,beta):
        """ Creates the structure of the model - store abilities

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
        scale, shape, skewness = self._get_scale_and_shape(parm)
        state_vectors = np.zeros(shape=(self.max_team+1))
        state_vectors_2 = np.zeros(shape=(self.max_team_2+1))
        state_vectors_store_1 = np.zeros(shape=(int(np.max(self.home_count)+50),int(self.max_team+1)))
        state_vectors_store_2 = np.zeros(shape=(int(np.max(self.home_2_count)+50),int(self.max_team_2+1)))
        theta = np.zeros(shape=(self.data.shape[0]))

        for t in range(0,self.data.shape[0]):
            theta[t] = parm[0] + state_vectors_2[self.home_2_id[t]] - state_vectors_2[self.away_2_id[t]] + state_vectors[self.home_id[t]] - state_vectors[self.away_id[t]]

            state_vectors[self.home_id[t]] += parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors[self.away_id[t]] += -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_2[self.home_2_id[t]] += parm[2]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_2[self.away_2_id[t]] += -parm[2]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)

            state_vectors_store_1[int(self.home_count[t]), self.home_id[t]] = state_vectors_store_1[max(0,int(self.home_count[t])-1), self.home_id[t]] + parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_store_1[int(self.away_count[t]), self.away_id[t]] = state_vectors_store_1[max(0,int(self.away_count[t])-1), self.away_id[t]] -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_store_2[int(self.home_2_count[t]), self.home_2_id[t]] = state_vectors_store_2[max(0,int(self.home_2_count[t])-1), self.home_2_id[t]] + parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)
            state_vectors_store_2[int(self.away_2_count[t]), self.away_2_id[t]] = state_vectors_store_2[max(0,int(self.away_2_count[t])-1), self.away_2_id[t]] -parm[1]*self.family.score_function(self.data[t], self.link(theta[t]), scale, shape, skewness)

        return state_vectors_store_1, state_vectors_store_2

    def add_second_component(self, team_1, team_2):
        self.home_2_id, self.away_2_id = self._create_ids(self.original_dataframe[team_1].values,self.original_dataframe[team_2].values)
        self.team_strings_2 = sorted(list(set(np.append(self.original_dataframe[team_1].values,self.original_dataframe[team_2].values))))
        self.team_dict_2 = dict(zip(self.team_strings_2, range(len(self.team_strings_2))))
        self.home_2_count, self.away_2_count = self._match_count_2()
        self.max_team_2 = max(np.max(self.home_2_id),np.max(self.away_2_id))

        self.z_no += 1
        self.latent_variables.z_list = []

        self.latent_variables.add_z('Constant', fam.Normal(0, 10,transform=None), fam.Normal(0,3))
        self.latent_variables.add_z('Ability Scale 1', fam.Normal(0,1,transform=None), fam.Normal(0,3))
        self.latent_variables.add_z('Ability Scale 2', fam.Normal(0,1,transform=None), fam.Normal(0,3))

        for no, i in enumerate(self.family.build_latent_variables()):
            self.latent_variables.add_z(i[0],i[1],i[2])
            self.latent_variables.z_list[3+no].start = i[3]
        self.latent_variables.z_list[0].start = self.mean_transform(np.mean(self.data))

        self._model = self._model_two_components
        self._model_abilities = self._model_abilities_two_components
        self.plot_abilities = self.plot_abilities_two_components
        self.predict = self.predict_two_components

    def neg_loglik(self, beta):
        theta, Y, _ = self._model(beta)
        parm = np.array([self.latent_variables.z_list[k].prior.transform(beta[k]) for k in range(beta.shape[0])])
        model_scale, model_shape, model_skewness = self._get_scale_and_shape(parm)
        return self.family.neg_loglikelihood(Y,self.link(theta),model_scale,model_shape,model_skewness)

    def plot_abilities_one_components(self, team_ids, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(15,5))

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            plt.figure(figsize=figsize)

            if type(team_ids) == type([]):
                if type(team_ids[0]) == str:
                    for team_id in team_ids:
                        plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[self.team_dict[team_id]],
                            trim='b'), label=self.team_strings[self.team_dict[team_id]])
                else:
                    for team_id in team_ids:
                        plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[team_id],
                            trim='b'), label=self.team_strings[team_id])
            else:
                if type(team_ids) == str:
                    plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[self.team_dict[team_ids]],
                        trim='b'), label=self.team_strings[self.team_dict[team_ids]])
                else:
                    plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[team_ids],
                        trim='b'), label=self.team_strings[team_ids])

            plt.legend()
            plt.ylabel("Power")
            plt.xlabel("Games")
            plt.show()

    def plot_abilities_two_components(self, team_ids, component_id=0, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = kwargs.get('figsize',(15,5))

        if component_id == 0:
            name_strings = self.team_strings
            name_dict = self.team_dict
        else:
            name_strings = self.team_strings_2
            name_dict = self.team_dict_2

        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            plt.figure(figsize=figsize)

            if type(team_ids) == type([]):
                if type(team_ids[0]) == str:
                    for team_id in team_ids:
                        plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[component_id].T[name_dict[team_id]],
                            trim='b'), label=name_strings[name_dict[team_id]])
                else:
                    for team_id in team_ids:
                        plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[component_id].T[team_id],
                            trim='b'), label=name_strings[team_id])
            else:
                if type(team_ids) == str:
                    plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[component_id].T[name_dict[team_ids]],
                        trim='b'), label=name_strings[name_dict[team_ids]])
                else:
                    plt.plot(np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[component_id].T[team_ids],
                        trim='b'), label=name_strings[team_ids])
            plt.legend()
            plt.ylabel("Power")
            plt.xlabel("Games")
            plt.show()

    def predict_one_component(self, team_1, team_2, neutral=False):
        """
        Returns team 1's probability of winning
        """
        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            if type(team_1) == str:
                team_1_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[self.team_dict[team_1]], trim='b')[-1]
                team_2_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[self.team_dict[team_2]], trim='b')[-1]
 
            else:
                team_1_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[team_1], trim='b')[-1]
                team_2_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values()).T[team_2], trim='b')[-1]

        t_z = self.transform_z()

        if neutral is False:
            return self.link(t_z[0] + team_1_ability - team_2_ability)
        else:
            return self.link(team_1_ability - team_2_ability)

    def predict_two_components(self, team_1, team_2, team_1b, team_2b, neutral=False):
        """
        Returns team 1's probability of winning
        """
        if self.latent_variables.estimated is False:
            raise Exception("No latent variables estimated!")
        else:
            if type(team_1) == str:
                team_1_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[0].T[self.team_dict[team_1]], trim='b')[-1]
                team_2_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[0].T[self.team_dict[team_2]], trim='b')[-1]
                team_1_b_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[1].T[self.team_dict[team_1]], trim='b')[-1]
                team_2_b_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[1].T[self.team_dict[team_2]], trim='b')[-1]
  
            else:
                team_1_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[0].T[team_1], trim='b')[-1]
                team_2_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[0].T[team_2], trim='b')[-1]
                team_1_b_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[1].T[team_1_b], trim='b')[-1]
                team_2_b_ability = np.trim_zeros(self._model_abilities(self.latent_variables.get_z_values())[1].T[team_2_b], trim='b')[-1]

        t_z = self.transform_z()

        if neutral is False:
            return self.link(t_z[0] + team_1_ability - team_2_ability + team_1_b_ability - team_2_b_ability)
        else:
            return self.link(team_1_ability - team_2_ability + team_1_b_ability - team_2_b_ability)

