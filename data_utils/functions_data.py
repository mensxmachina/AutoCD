
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Functions for data preprocessing
# Author: kbiza@csd.uoc.gr


def get_data_type(data_pd):

    '''
    Returns the type of each variable in the dataset (continuous or categorical)
    Parameters
    ----------
        data_pd(pandas dataframe): the dataset

    Returns
    -------
        data_type_info(pandas dataframe): with two columns,
                        - 'var_type' for the type of the variable
                        - 'n_domain' for the number of categories in case of categorical variable
                        * each row corresponds to a variable
    '''

    unique_val_thr = 5

    d = {'var_type': ['continuous' for i in data_pd.columns],
         'n_domain': [0 for i in data_pd.columns]}
    data_type_info = pd.DataFrame(data=d, index=data_pd.columns)

    for var in data_pd.columns:
        cur_col = pd.to_numeric(data_pd[var], errors='coerce')  # check if the column has only str
        if pd.isna(cur_col).all():  # input is str
            data_type_info.loc[var, 'var_type'] = 'categorical'
            data_type_info.loc[var, 'n_domain'] = np.unique(data_pd[var]).shape[0]
        else:
            if len(data_pd[var].unique()) < unique_val_thr:  # number of unique values < threshold
                # print('cat var', var, len(data_pd[var].unique()))
                data_type_info.loc[var, 'var_type'] = 'categorical'
                unique_classes = data_pd[var].unique()
                # print(unique_classes)
                # data_type_info.loc[var, 'n_domain'] = np.nanmax(unique_classes) + 1
                data_type_info.loc[var, 'n_domain'] = len(data_pd[var].unique())

    return data_type_info

def apply_ordinal_encoding(data_pd, data_type_info):

    '''
    Applies ordinal encoding with sklearn
    Parameters
    ----------
        data_pd(pandas dataframe): the dataset
        data_type_info(pandas dataframe): information for the type of each variable (continuous or categorical)

    Returns
    -------
        data_pd(pandas dataframe):  the transformed dataset
    '''

    categorical_var_names = data_type_info.index[data_type_info['var_type'] == 'categorical']  # .tolist()
    ord_encoder = OrdinalEncoder()
    ord_encoder.fit(data_pd[categorical_var_names])
    data_pd[categorical_var_names] = ord_encoder.transform(data_pd[categorical_var_names])

    return data_pd


def timeseries_to_timelagged(data_pd, n_lags, window=False):

    '''
    Converts time-series data to time-lagged data
    Parameters
    ----------
        data_pd (pandas dataframe): time-series dataset
                    e.g. V1, V2
        n_lags(int) : number fo previous lags
        window(bool) : True for non-overlapped windows

    Returns
    -------
        data_pd_tl(pandas dataframe) : time-lagged dataset
                    e.g. V1, V2, V1:1, V2:1
    '''

    n_samples = data_pd.shape[0]
    n_nodes = data_pd.shape[1]
    T = n_lags + 1
    var_names = np.empty((1, n_nodes * T), dtype='O')
    data_tl = np.empty((n_samples - n_lags, n_nodes * T), dtype=float)

    for row in range(n_lags, n_samples):
        time_row = np.empty((1, n_nodes * T), dtype='O')
        c = 0
        for t in range(T):
            var_row_minus_t = data_pd.iloc[[row - t]].to_numpy()
            time_row[:, c:c + var_row_minus_t.shape[1]] = var_row_minus_t
            c += var_row_minus_t.shape[1]

        data_tl[row - n_lags, :] = time_row

    # Names
    c = 0
    for t in range(T):
        if t == 0:
            cur_t_name = data_pd.columns
        else:
            cur_t_name = data_pd.columns + ':' + str(t)
        var_names[0, c:c + cur_t_name.shape[0]] = cur_t_name
        c += cur_t_name.shape[0]

    var_names = var_names.reshape(-1, )

    data_pd_tl = pd.DataFrame(data_tl, columns=var_names)

    if window:
        step_ = n_lags + 1
        idx = np.arange(0, data_pd_tl.shape[0], step=step_)
        data_pd_tl = data_pd_tl.iloc[idx, :].reset_index(inplace=False, drop=True)

    return data_pd_tl


def timelagged_to_timeseries(data_pd, n_lags):

    '''
    Converts time-lagged data to time-series data
    Parameters
    ----------
        data_pd(pandas dataframe): time-lagged dataset
                e.g. V1, V2, V1:1, V2:1
        n_lags(int) : number fo previous lags

    Returns
    -------
        ts_data(pandas dataframe) : time-series dataset
                e.g. V1, V2
    '''

    T = n_lags + 1
    n_nodes = int(data_pd.shape[1] / T)
    n_rows = data_pd.shape[0] * T
    data_tseries = np.zeros((n_rows, n_nodes), dtype=float)

    for row_lg in range(data_pd.shape[0]):
        for node in range(n_nodes):
            for lag, lag_rv in zip(range(T), reversed(range(T))):
                column_lg = n_nodes * lag_rv + node
                c = T * row_lg + lag
                data_tseries[c, node] = data_pd.iloc[row_lg, column_lg]

    data_tseries_pd = pd.DataFrame(data_tseries, columns = data_pd.columns[0:n_nodes])
    return data_tseries_pd

def logMp(xpd):
    '''
    For data transformation with sklearn
    '''

    NegCols = xpd.columns[(xpd < 0).any()]
    PosCols = np.setdiff1d(xpd.columns, NegCols)

    Xpd2 = pd.DataFrame(columns=xpd.columns)
    Xpd2[NegCols] = xpd[NegCols] - xpd[NegCols].min() + 1
    Xpd2[PosCols] = xpd[PosCols] + xpd[PosCols].min() + 1

    Xpd2 = np.log(Xpd2)
    return Xpd2

def transform_data(data_pd, data_type_info, transform_type):

    '''
    Data transfomation with sklearn
    Parameters
    ----------
        data_pd(pandas dataframe): dataset
        data_type_info(pandas dataframe)
        transform_type(str):{qgaussian, log, minmax, standardize}

    Returns
    -------
        transformed_data(pandas dataframe)
    '''

    transformed_data = data_pd.copy()
    continuous_var_names = data_type_info.index[data_type_info['var_type'] == 'continuous']

    if transform_type == 'qgaussian':
        qt = preprocessing.QuantileTransformer(output_distribution='normal')
        transformed_data[continuous_var_names] = qt.fit_transform(data_pd[continuous_var_names])

    elif transform_type == 'log':
        logt = preprocessing.FunctionTransformer(logMp)
        transformed_data[continuous_var_names] = logt.transform(data_pd[continuous_var_names])

    elif transform_type == 'minmax':
        minmaxt = MinMaxScaler()
        transformed_data[continuous_var_names] = minmaxt.fit_transform(data_pd[continuous_var_names])

    elif transform_type == 'standardize':
        stdt = StandardScaler()
        transformed_data[continuous_var_names] = stdt.fit_transform(data_pd[continuous_var_names])

    return transformed_data

