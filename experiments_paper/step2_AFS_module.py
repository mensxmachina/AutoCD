
import jpype
from jpype import *
from jpype.types import *
import jpype.imports
import json
import pickle
import time

from AutoCD.predictive_modeling.class_predictive_configurator import *
from AutoCD.predictive_modeling.class_AFS import *
from AutoCD.predictive_modeling.class_predictive_model import *
from AutoCD.data_utils.functions_variable_names import *
from AutoCD.data_utils.class_data_object import *
from AutoCD.metrics_evaluation.evaluate_prec_rec_sets import *


# ----------------------------------------------------------------------------------------------------------
# Experiment: Apply the AFS module on the simulated temporal data to reduce the dimensionality of the problem
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------------------

def main():

    # file names
    path = './files_results/'
    id = '20n_'                 # change the file name if needed
    input_name = path + id + 'files.pkl'
    params_name = path + id + 'params.pkl'
    output_name = path + id + 'files_mb.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    # create predictive configurations
    pred_confirator = PredictiveConfigurator()
    pred_configs = pred_confirator.create_predictive_configs()
    params['predictive_configs'] = pred_configs

    # experiment
    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        target_name = files[rep]['target_name']
        true_mb_names = files[rep]['true_mb_names']
        true_mb_nameslag = files[rep]['true_mb_nameslag']
        data_train = files[rep]['data_train']
        data_test = files[rep]['data_test']
        true_pred_config = files[rep]['true_pred_config']

        dataObj = data_object(data_train, 'train', target_name, params['n_lags'])
        est_mb_idx, est_mb_names_lag, max_avg_pred, model, opt_pred_config = AFS().run_AFS(pred_configs, dataObj)
        est_mb_names = names_from_lag(est_mb_names_lag)
        prec_mb, rec_mb = evaluate_prec_rec_sets(true_mb_names, est_mb_names)

        study_vars_ = np.concatenate((est_mb_names, [target_name]))
        study_vars_ = list(set(study_vars_))
        study_vars = [s for s in dataObj.samples.columns if s in study_vars_]

        dataObj.samples_all = dataObj.samples.copy()
        dataObj.samples = dataObj.samples[study_vars].copy()
        dataObj.data_type_info_all = dataObj.data_type_info.copy()
        dataObj.data_type_info = dataObj.data_type_info.loc[study_vars].copy()


        # Predictive performance on Dtest

        data_train_lagged = timeseries_to_timelagged(data_train, n_lags=params['n_lags'])
        data_test_lagged = timeseries_to_timelagged(data_test, n_lags=params['n_lags'])

        pm = Predictive_Model()
        true_model, y_test_truemb, _= pm.predictive_modeling(true_pred_config, 'continuous',
                                            data_train_lagged[true_mb_nameslag].to_numpy(),
                                            data_train_lagged[target_name].to_numpy(),
                                            data_test_lagged[true_mb_nameslag].to_numpy())
        y_test_estmb = model.predict(data_test_lagged[est_mb_names_lag].to_numpy())

        r2_truemb = r2_score(data_test_lagged[target_name], y_test_truemb)
        r2_estmb = r2_score(data_test_lagged[target_name], y_test_estmb)

        files[rep]['dataObj'] = dataObj
        files[rep]['est_mb_names'] = est_mb_names
        files[rep]['est_mb_names_lag'] = est_mb_names_lag
        files[rep]['prec_mb'] = prec_mb
        files[rep]['rec_mb'] = rec_mb
        files[rep]['dataObj'] = dataObj
        files[rep]['model'] = model
        files[rep]['opt_pred_config'] = opt_pred_config
        files[rep]['r2_estmb'] = r2_estmb
        files[rep]['r2_truemb'] = r2_truemb
        files[rep]['deltar2'] = r2_truemb - r2_estmb
        files[rep]['afs_exec_time'] = time.time() - t0

    precs_mb = [d.get('prec_mb') for d in files]
    recs_mb = [d.get('rec_mb') for d in files]
    delta_r2s = [d.get('deltar2') for d in files]
    print('mean Prec:%0.2f' %(np.mean(precs_mb)), 'SE:%0.2f' %(np.std(precs_mb) / np.sqrt(len(precs_mb))))
    print('mean Rec:%0.2f' %(np.mean(recs_mb)), 'SE:%0.2f' %(np.std(recs_mb) / np.sqrt(len(recs_mb))))
    print('mean DeltaR2:%0.2f' %(np.mean(delta_r2s)), 'SE:%0.2f' %(np.std(delta_r2s) / np.sqrt(len(delta_r2s))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

