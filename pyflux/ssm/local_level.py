from .. import families as fam
from .. import tsm as tsm

from .llm import *
from .nllm import *

class LocalLevel(tsm.TSM):
    """ Wrapper for local level models

    **** LOCAL LEVEL MODEL ****

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.

    integ : int (default : 0)
        Specifies how many times to difference the time series.

    target : str (pd.DataFrame) or int (np.array)
        Specifies which column name or array index to use. By default, first
        column/array will be selected as the dependent variable.

    family : 
        e.g. pf.Normal(0,1)
    """

    def __new__(cls, data, family, integ=0, target=None):
        if isinstance(family, fam.Normal):
            return LLEV(data=data, integ=integ, target=target)
        else:
            return NLLEV(data=data, family=family, integ=integ, target=target)

    def __init__(self, data, family, integ=0, target=None):
        pass

