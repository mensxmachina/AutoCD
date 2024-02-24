
from sklearn.utils import resample
from AutoCD.causal_discovery.functions_causal_discovery import *
from AutoCD.metrics_evaluation.edge_metrics_bootstraps import *


def CRV_module_confidence(dataObj, opt_causal_config, mec_causal_graph, B=50, n_lags=None):

    '''
    Computes the edge confidences using bootstrapping
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        dataObj(class object) : contains the dataset and information regarding the data
        opt_causal_config(dictionary): the selected causal configuration
        mec_causal_graph(pandas Dataframe): the estimated causal graph
        B(int): number of bootstraps
        n_lags(int or None): the maximum number of previous time lags in case of a time-lagged graph

    Returns
    -------
        edge_confidences(pandas Dataframe): the estimated edge consistency and edge discovery frequencies
    '''


    # Create bootstrapped samples
    if isinstance(n_lags, int):
        window_resampling = 2 * n_lags
        samples_for_boostrap = timeseries_to_timelagged(dataObj.samples, window_resampling, window=True)
    else:
        samples_for_boostrap = dataObj.samples.copy()


    # Apply the selected causal configuration on the bootstrapped samples
    bootstrapped_matrix_mec = []
    b = 0
    while b < B:
        bootstrapped_samples_ = resample(samples_for_boostrap,
                                         n_samples=samples_for_boostrap.shape[0], replace=True)

        if isinstance(n_lags, int):
            is_time_series = True
            bootstrapped_samples = timelagged_to_timeseries(bootstrapped_samples_, window_resampling)
        else:
            is_time_series = False
            bootstrapped_samples = bootstrapped_samples_

        library_results, boost_mec_graph, boost_graph = \
            causal_discovery(opt_causal_config, bootstrapped_samples, dataObj.data_type_info, is_time_series)

        if isinstance(boost_mec_graph, pd.DataFrame):
            bootstrapped_matrix_mec.append(boost_mec_graph)
            b += 1

    # Compute edge consistency and discovery frequency on the bootstrapped graphs
    edge_confidences, matrix_confidences = edge_metrics_on_bootstraps(
        best_mec_matrix=mec_causal_graph, bootstrapped_mec_matrix=bootstrapped_matrix_mec)

    return edge_confidences