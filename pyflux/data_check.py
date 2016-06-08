import pandas as pd
import numpy as np

def data_check(data,target):
    """ Checks data type

    Parameters
    ----------
    data : pd.DataFrame or np.array
        Field to specify the time series data that will be used.
    
    target : int or str
        Target column
        
    Returns
    ----------
    transformed_data : np.array
        Raw data array for use in the model
        
    data_name : str
        Name of the data
    
    is_pandas : Boolean
        True if pandas data, else numpy
    
    data_index : np.array
        The time indices for the data
    """

    # Check pandas or numpy
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.core.frame.DataFrame):
        data_index = data.index         
        if target is None:
            transformed_data = data.ix[:,0].values
            data_name = str(data.columns.values[0])
        else:
            transformed_data = data[target].values          
            data_name = str(target)                 
        is_pandas = True
        
    elif isinstance(data, np.ndarray):
        data_name = "Series"        
        is_pandas = False
        if any(isinstance(i, np.ndarray) for i in data):
            if target is None:
                transformed_data = data[0]          
                data_index = list(range(len(data[0])))
            else:
                transformed_data = data[target]         
                data_index = list(range(len(data[target])))
        else:
            transformed_data = data                 
            data_index = list(range(len(data)))
    else:
        raise Exception("The data input is not pandas or numpy compatible!")
    
    return transformed_data, data_name, is_pandas, data_index

def mv_data_check(data,check):
    # Check pandas or numpy

    if isinstance(data, pd.DataFrame):
        data_index = data.index     
        transformed_data = data.values
        data_name = data.columns.values
        is_pandas = True

    elif isinstance(data, np.ndarray):
        data_name = np.asarray(range(1,len(data[0])+1)) 
        is_pandas = False
        transformed_data = data
        data_index = list(range(len(data[0])))

    else:
        raise Exception("The data input is not pandas or numpy compatible!")

    return transformed_data, data_name, is_pandas, data_index
