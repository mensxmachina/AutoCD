
from AutoCD.predictive_modeling.class_predictive_configurator import *
from AutoCD.predictive_modeling.class_AFS import *
from AutoCD.data_utils.functions_variable_names import *
from AutoCD.data_utils.class_data_object import *

def AFS_module(data_pd, target_name, n_lags=None):
    '''

    Applies the AFS module
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        data_pd (pandas Dataframe): the dataset
        target_name (str) : the name of the target
        n_lags(int or None): the number of time lags in case of time series data

    Returns
    -------
        est_mb_names (list) :names of the selected features (markov boundary) of the target node
        model (sklearn model): the selected predictive model
        max_avg_pred (float): the average out-of-sample predictive performance of the selected model
        dataObj (class object) : contains the preprocessed reduced data
    '''


    # create predictive configurations
    pred_confirator = PredictiveConfigurator()
    pred_configs = pred_confirator.create_predictive_configs()

    # data preprocessing
    #   currently AFS supports basic data transformations
    data_type = get_data_type(data_pd)
    transformed_data = transform_data(data_pd, data_type, 'standardize')

    dataObj = data_object(transformed_data,'AutoCD_data', target_name, n_lags)
    est_mb_idx, est_mb_names_, max_avg_pred, model, opt_pred_config = AFS().run_AFS(pred_configs, dataObj)

    if isinstance(n_lags, int):
        est_mb_names = names_from_lag(est_mb_names_)
    else:
        est_mb_names = est_mb_names_

    # Reduce the dataset
    study_vars_ = np.concatenate((est_mb_names, [target_name]))
    study_vars_ = list(set(study_vars_))
    study_vars = [s for s in dataObj.samples.columns if s in study_vars_]

    dataObj.samples = dataObj.samples[study_vars].copy()
    dataObj.data_type_info = dataObj.data_type_info.loc[study_vars].copy()

    return est_mb_names, model, max_avg_pred, dataObj