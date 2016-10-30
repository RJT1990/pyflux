from .. import families as fam
from .. import tsm as tsm

from .llt import *
from .nllt import *

class LocalTrend(tsm.TSM):
    """ Wrapper for local linear trend models

    **** LOCAL LINEAR TREND MODEL ****

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
            return LLT(data=data, integ=integ, target=target)
        else:
            return NLLT(data=data, family=family, integ=integ, target=target)

    def __init__(self, data, family, integ=0, target=None):
        pass

