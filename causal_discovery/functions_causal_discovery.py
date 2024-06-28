
from AutoCD.causal_discovery.class_Tetrad import *
from AutoCD.causal_discovery.class_Tigramite import *
from AutoCD.data_utils.functions_data import *


def causal_discovery(configuration, samples, data_type_info, is_time_series, tiers=None):

    '''
    Causal discovery with a given a causal configuration
    Author : kbiza@csd.uoc.gr

    Parameters
    ----------
        configuration (dictionary) : the causal configuration to run
        samples (pandas Dataframe) :  the dataset
        data_type_info (pandas Dataframe) : information about the type of each variable (continuous or categorical)
        is_time_series (bool) : True if dataset contains temporal variables

    Returns
    -------
        library_results (dictionary):
            the causal output as returned from the causal discovery package (tetrad or tigramite)
        mec_graph_pd (pandas Dataframe) :
            the estimated Markov Equivalence Class (PDAG or PAG)
        graph_pd (pandas Dataframe) :
            a causal graph (DAG or MAG) from the estimated MEC

        Note : in case of temporal data the causal graph is a time-lagged causal graph

        In any case (temporal or cross sectional data)
            mec_graph_pd and graph_pd are matrices with the following notation:
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs or i---j in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs
    '''

    tetrad = ['pc', 'cpc', 'fges', 'directlingam',
             'fci', 'fcimax', 'rfci', 'cfci', 'gfci', 'svarfci', 'svargfci']
    tigramite = ['PCMCI', 'PCMCI+', 'LPCMCI']

    if is_time_series:
        # all output graphs will be time lagged graphs
        # tetrad needs time-lagged dataset

        # we follow the method appeared in tetrad to create lagged datasets (i.e. we do not use window=True)
        samples_tetrad = timeseries_to_timelagged(samples, configuration['n_lags'])

        var_names_lagged = samples_tetrad.columns
        # data_type_info_tetrad = get_data_type(samples_tetrad)
        data_type_info_lagged=[]
        index_rows = []
        for lag in range(configuration['n_lags']+1):
            data_type_info_lagged.append(data_type_info)
            if lag==0:
                index_rows = index_rows+ data_type_info.index.to_list()
            else:
                index_rows_= [s+':'+str(lag) for s in data_type_info.index]
                index_rows += index_rows_
        data_type_info_tetrad = pd.concat(data_type_info_lagged)
        data_type_info_tetrad.index = index_rows

    else:
        samples_tetrad = samples
        var_names_lagged = None
        data_type_info_tetrad = data_type_info

    print('\trun causal discovery with ', configuration['name'])
    if configuration['name'] in tetrad:
        alg = Tetrad(samples_tetrad, data_type_info_tetrad, is_time_series, tiers)

    elif configuration['name'] in tigramite:
        alg = Tigramite(samples, var_names_lagged, data_type_info)

    else:
        raise ValueError('%s config not included' % configuration['name'])

    library_results, mec_graph_pd, graph_pd = alg.run(configuration)

    return library_results, mec_graph_pd, graph_pd