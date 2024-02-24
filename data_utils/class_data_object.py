
from AutoCD.data_utils.functions_data import *


class data_object():

    '''
    Class for data utils
    Author: kbiza@csd.uoc.gr
    '''

    def __init__(self, data_pd,  dataset_name, target_name=None, n_lags=None):

        '''
        Creates a data object
        Parameters
        ----------
            data_pd(pandas dataframe:  the dataset (not time-lagged data)
            dataset_name(str): the name of the dataset
            target_name(str):  the name of the target , as appears in the columns of data_pd
            n_lags(int): number of previous time lags in case of time-series data
        '''

        # extract data type information
        self.data_type_info = get_data_type(data_pd)

        # apply ordinal encoding
        self.samples = apply_ordinal_encoding(data_pd, self.data_type_info)

        if self.data_type_info['var_type'].eq('continuous').all():
            self.data_type = 'continuous'
        elif self.data_type_info['var_type'].eq('categorical').all():
            self.data_type = 'categorical'
        else:
            self.data_type = 'mixed'

        # target variable information
        if target_name:
            self.target_info={}
            self.target_info['name'] = target_name
            self.target_info['var_type'] = self.data_type_info['var_type'].loc[target_name]

        # dataset_name
        self.dataset_name = dataset_name

        # time info
        if isinstance(n_lags, int):
            self.nlags = n_lags
            self.is_time_series = True
        else:
            self.is_time_series = False





