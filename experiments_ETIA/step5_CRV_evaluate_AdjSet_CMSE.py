
import pickle
from AutoCD.predictive_modeling.class_AFS import *
from AutoCD.metrics_evaluation.evaluate_prec_rec_sets import *

# ----------------------------------------------------------------------------------------------------------
# Experiment: Evaluate the identified adjustment sets
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------------------


def main():

    path = './files_results/'
    id = '100n_2500s_3ad_6md_1exp_1rep_'                 # change the file name if needed
    input_name = path + id + 'files_mb_cd_adjset.pkl'
    output_name = path + id + 'files_mb_cd_adjset_cmse_minimal.pkl'

    results_counts_name = path + id + 'counts_res_adjset.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    agree_identical = 0
    agree_different = 0
    disagree = 0

    counts_res = {}
    for rep in range(len(files)):

        print('\n Rep %d' %rep)

        Z_est = files[rep]['Z_est_min_pag']
        Z_true = files[rep]['Z_true_min_dag']

        if not isinstance(Z_est, list) and not isinstance(Z_true, list):
            agree_identical += 1

        elif isinstance(Z_est, list) and isinstance(Z_true, list):
            if set(Z_true) == set(Z_est):
                agree_identical += 1
            else:
                agree_different += 1

        else:
            disagree += 1

        data_train = files[rep]['data_train']
        target_name = files[rep]['target_name']
        exposure_names = files[rep]['exposure_names']

        pred_config={}
        pred_config['pred_name'] = 'linear_regression'

        if isinstance(Z_true, list):
            exposures_Z_true = np.concatenate((exposure_names, Z_true))
            pm = Predictive_Model()
            true_model_EtrueZ, _, _ = pm.predictive_modeling(pred_config, 'continuous',
                                                                  data_train[exposures_Z_true].to_numpy(),
                                                                  data_train[target_name].to_numpy())

            true_beta_E = true_model_EtrueZ.coef_[0] # here exposure is the first var

        else:
            true_beta_E = 0


        if isinstance(Z_est, list):
            exposures_Z_est = np.concatenate((exposure_names, Z_est))

            pm = Predictive_Model()
            est_model_EestZ, _, _ = pm.predictive_modeling(pred_config, 'continuous',
                                                            data_train[exposures_Z_est].to_numpy(),
                                                            data_train[target_name].to_numpy())

            est_beta_E = est_model_EestZ.coef_[0] # here exposure is the first var

        else:
            est_beta_E = 0

        files[rep]['true_beta_E'] = true_beta_E
        files[rep]['est_beta_E'] = est_beta_E
        files[rep]['beta_squared_dif'] = np.sqrt(((true_beta_E- est_beta_E)**2 ))

    counts_res['agree_identical'] = agree_identical / len(files)
    counts_res['agree_different'] = agree_different / len(files)
    counts_res['disagree'] = disagree/ len(files)

    beta_squared_dif_ = [d.get('beta_squared_dif') for d in files]
    print('median beta dif:%0.2f' % (np.median(beta_squared_dif_)), 'SE:%0.2f' % (np.std(beta_squared_dif_) / np.sqrt(len(beta_squared_dif_))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)

    with open(results_counts_name, 'wb') as outp:
        pickle.dump(counts_res, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()