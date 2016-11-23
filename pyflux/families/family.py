import numpy as np

class Family(object):

    def __init__(self, transform=None, **kwargs):
        """
        Parameters
        ----------
        transform : str
            Whether to apply a transformation - e.g. 'exp' or 'logit'
        """
        self.transform_name = transform     
        self.transform = self.transform_define(transform)
        self.itransform = self.itransform_define(transform)
        self.itransform_name = self.itransform_name_define(transform)

    @staticmethod
    def ilogit(x):
        return 1.0/(1.0+np.exp(-x))

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1.0 - x)

    @staticmethod
    def transform_define(transform):
        """
        This function links the user's choice of transformation with the associated numpy function
        """
        if transform == 'tanh':
            return np.tanh
        elif transform == 'exp':
            return np.exp
        elif transform == 'logit':
            return Family.ilogit
        elif transform is None:
            return np.array
        else:
            return None

    @staticmethod
    def itransform_define(transform):
        """
        This function links the user's choice of transformation with its inverse
        """
        if transform == 'tanh':
            return np.arctanh
        elif transform == 'exp':
            return np.log
        elif transform == 'logit':
            return Family.logit
        elif transform is None:
            return np.array
        else:
            return None

    @staticmethod
    def itransform_name_define(transform):
        """
        This function is used for model results table, displaying any transformations performed
        """
        if transform == 'tanh':
            return 'arctanh'
        elif transform == 'exp':
            return 'log'
        elif transform == 'logit':
            return 'ilogit'
        elif transform is None:
            return ''
        else:
            return None
