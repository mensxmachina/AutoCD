
import jpype
from jpype import *
from jpype.types import *
import jpype.imports
import json
import pickle
import time

from AutoCD.predictive_modeling.class_predictive_configurator import *
from AutoCD.predictive_modeling.class_AFS import *
from AutoCD.data_utils.class_data_object import *
from AutoCD.metrics_evaluation.evaluate_prec_rec_sets import *

# ----------------------------------------------------------------------------------------------------------
# Experiment: Apply the AFS module on the simulated data to reduce the dimensionality of the problem
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------------------


def main():

    # file names
    path = './files_results/'
    id = '100n_2500s_3ad_6md_1exp_1rep_'           # change the file name if needed
    input_name = path + id + 'files.pkl'
    params_name = path + id + 'params.pkl'
    output_name = path + id + 'files_mb.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        target_name = files[rep]['target_name']
        data_train = files[rep]['data_train']
        data_test = files[rep]['data_test']
        exposure_names = files[rep]['exposure_names']
        true_MB_names = files[rep]['true_MB_names']
        target_idx = files[rep]['target_idx']
        exposure_idx = files[rep]['exposure_idx']
        true_pred_config = files[rep]['true_pred_config']
        true_mb_target = files[rep]['true_mb_target']

        # AFS on the target
        dataObj = data_object(data_train, id, target_name)

        # create predictive configurations
        pred_confirator = PredictiveConfigurator()
        pred_configs = pred_confirator.create_predictive_configs(dataObj)
        params['predictive_configs'] = pred_configs

        (est_mb_names_idx, est_mb_names_target, _,
         opt_pred_model_target_given_mb, opt_pred_config_target_given_mb) = AFS().run_AFS(pred_configs, dataObj)

        # AFS on each Mb member of the target
        set_idx_ = est_mb_names_idx
        for mb_name in est_mb_names_target:
            dataObj_ = data_object(data_train, id, mb_name)
            mb_idx_, _, _, _, _ = AFS().run_AFS(pred_configs, dataObj_)
            set_idx_ = set_idx_ + mb_idx_
        set_t_mb_idx = list(set(set_idx_))
        set_t_mb_idx.sort()

        # AFS on each exposure
        set_idx_ = []
        for e_name in exposure_names:
            dataObj_ = data_object(data_train, id, e_name)
            mb_idx_, _, _, _, _ = AFS().run_AFS(pred_configs, dataObj_)
            set_idx_ = set_idx_ + mb_idx_
        set_exp_mb_idx = list(set(set_idx_))
        set_exp_mb_idx.sort()

        # Evaluate MB identification (Mb(target) , Mb(exposures))
        est_MB_idx = list(set(set_t_mb_idx + set_exp_mb_idx))
        est_MB_idx.sort()
        est_MB_names = data_train.columns[est_MB_idx]
        prec_mb, rec_mb = evaluate_prec_rec_sets(true_MB_names, est_MB_names)

        # Study vars : target + exposure + mb
        est_study_idx = list(set([target_idx] + exposure_idx + set_t_mb_idx + set_exp_mb_idx))
        est_study_idx.sort()
        est_study_names = data_train.columns[est_study_idx]

        dataObj.samples_all = dataObj.samples.copy()
        dataObj.samples = dataObj.samples[est_study_names].copy()
        dataObj.data_type_info_all = dataObj.data_type_info.copy()
        dataObj.data_type_info = dataObj.data_type_info.loc[est_study_names].copy()

        # Predictive performance on Dtest
        pm = Predictive_Model()
        true_model_target_given_mb, y_test_truemb, _ = pm.predictive_modeling(true_pred_config, 'continuous',
                                                              data_train[true_mb_target].to_numpy(),
                                                              data_train[target_name].to_numpy(),
                                                              data_test[true_mb_target].to_numpy())
        y_test_estmb = opt_pred_model_target_given_mb.predict(data_test[est_mb_names_target].to_numpy())

        r2_truemb = r2_score(data_test[target_name], y_test_truemb)
        r2_estmb = r2_score(data_test[target_name], y_test_estmb)

        files[rep]['dataObj'] = dataObj
        files[rep]['est_study_names'] = est_study_names
        files[rep]['est_study_idx'] = est_study_idx

        files[rep]['est_MB_names'] = est_MB_names
        files[rep]['est_MB_idx'] = est_MB_idx
        files[rep]['est_mb_names_target'] = est_mb_names_target

        files[rep]['prec_mb'] = prec_mb
        files[rep]['rec_mb'] = rec_mb
        files[rep]['r2_estmb'] = r2_estmb
        files[rep]['r2_truemb'] = r2_truemb
        files[rep]['deltar2'] = r2_truemb - r2_estmb
        files[rep]['opt_pred_config_target_given_mb'] = opt_pred_config_target_given_mb
        files[rep]['opt_pred_model_target_given_mb'] = opt_pred_model_target_given_mb
        files[rep]['afs_exec_time'] = time.time() - t0

    precs_mb = [d.get('prec_mb') for d in files]
    recs_mb = [d.get('rec_mb') for d in files]
    delta_r2s = [d.get('deltar2') for d in files]
    print('mean Prec:%0.2f' %(np.mean(precs_mb)), 'SE:%0.2f' %(np.std(precs_mb) / np.sqrt(len(precs_mb))))
    print('mean Rec:%0.2f' %(np.mean(recs_mb)), 'SE:%0.2f' %(np.std(recs_mb) / np.sqrt(len(recs_mb))))
    print('mean DeltaR2:%0.2f' % (np.mean(delta_r2s)), 'SE:%0.2f' % (np.std(delta_r2s) / np.sqrt(len(delta_r2s))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

