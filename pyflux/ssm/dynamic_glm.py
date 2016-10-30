from .. import families as fam
from .. import tsm as tsm

from .dynlin import *
from .ndynlin import *

class DynamicGLM(tsm.TSM):
    """ Wrapper for dynamic GLM models

    Parameters
    ----------
    formula : str
        Patsy string specifying the regression

    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    family : 
        e.g. pf.Normal(0,1)
    """

    def __new__(cls, formula, data, family):
        if isinstance(family, fam.Normal):
            return DynReg(formula=formula, data=data)
        else:
            return NDynReg(formula=formula, data=data, family=family)

    def __init__(self, formula, data, family):
        pass

