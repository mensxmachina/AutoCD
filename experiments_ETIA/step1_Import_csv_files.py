
import numpy as np
import json
import pickle

from AutoCD.data_utils.functions_data import *
from AutoCD.causal_graph_utils.markov_boundary import *
from AutoCD.causal_graph_utils.create_sub_mag_pag import *


# ------------------------------------------------------------------------------
# Experiment: Import csv files and create the necessary pkl files for the experiment
#              Change the number of nodes (params['n_nodes']= {100, 200, 500, 1000})
# Author: kbiza@csd.uoc.gr
# ------------------------------------------------------------------------------

def main():

    # Simulation parameters
    params = {}
    params['n_nodes'] = 100  # change the number of nodes {100, 200, 500, 1000}

    params['n_rep'] = 10                # number of repetitions
    params['n_samples'] = 2500          # total number of samples
    params['n_test'] = 500              # number of samples to hold out
    params['avg_degree'] = 3            # average node degree
    params['max_degree'] = 6            # maximum node degree
    params['simulation_type'] = 'GSEM'  # GSEM, BayesNet, LeeHastie, CondGauss,TimeSeries
    params['n_exposures'] = 1
    params['len_path_exp'] = 1
    params['len_parent_target'] = 1
    params['is_time_series'] = False
    params['n_lags'] = None

    csv_path = './csv_files/'+str(params['n_nodes'])+'nodes/'
    path = './files_results/'
    id_file = (str(params['n_nodes'] ) + 'n_' +
               str(params['n_samples']) + 's_' +
               str(params['avg_degree'])+ 'ad_' +
               str(params['max_degree']) + 'md_' +
               str(params['n_exposures']) + 'exp_'+
               str(params['n_rep']) + 'rep_'
               )
    files_name = path + id_file + 'files.pkl'
    params_name = path + id_file + 'params.pkl'

    files = [{} for _ in range(params['n_rep'])]

    for rep in range(params['n_rep']):

        dag_pd = pd.read_csv(csv_path+'true_dag_'+str(params['n_nodes'])+'_'+str(rep)+'.csv', index_col=0)
        _data_syn =pd.read_csv(csv_path+'data_' + str(params['n_nodes']) + '_' + str(rep) + '.csv', index_col=0)
        target_name_pd = pd.read_csv(csv_path+'target_name_' + str(params['n_nodes']) + '_' + str(rep) + '.csv', index_col=0)
        target_name = target_name_pd['target_name'][0]
        target_idx = dag_pd.columns.get_loc(target_name)

        exposure_name_pd = pd.read_csv(csv_path+'exposure_name_' + str(params['n_nodes']) + '_' + str(rep) + '.csv', index_col=0)
        exposure_name = exposure_name_pd['exposure_name'][0]
        exposure_idx = [dag_pd.columns.get_loc(exposure_name)]

        # Mb(target)
        true_dag = dag_pd.copy()
        true_mb_idx = markov_boundary(target_idx, true_dag)
        true_mb_target = dag_pd.columns[true_mb_idx]

        set_idx_ = true_mb_idx
        for t in true_mb_idx:
            set_idx_ = set_idx_ + markov_boundary(t, true_dag)
        set_t_mb_idx = list(set(set_idx_))
        set_t_mb_idx.sort()

        # Mb(exposures)
        set_idx_ = []
        for e in exposure_idx:
            set_idx_ = set_idx_ + markov_boundary(e, true_dag)
        set_exp_mb_idx = list(set(set_idx_))
        set_exp_mb_idx.sort()

        # True MB (of the target + exposures)
        true_MB_idx = list(set(set_t_mb_idx + set_exp_mb_idx))
        true_MB_idx.sort()
        true_MB_names = true_dag.columns[true_MB_idx]

        #Study vars : target + exposure + all mbs
        true_study_idx = list(set([target_idx] + exposure_idx + set_t_mb_idx + set_exp_mb_idx))
        true_study_idx.sort()
        true_study_names = true_dag.columns[true_study_idx]

        # True PAG
        true_mag_study, true_pag_study = create_sub_mag_pag(true_dag, true_study_names)

        # Transform data
        data_train = _data_syn.iloc[0:params['n_test'],:].copy()
        data_test = _data_syn.iloc[params['n_test']:-1, :].copy()

        data_type_train = get_data_type(data_train)
        data_type_test = get_data_type(data_test)
        transformed_train = transform_data(data_train, data_type_train, 'standardize')
        transformed_test = transform_data(data_test, data_type_test, 'standardize')

        # Based on the simulation function, we assume that a linear regression model is the true model
        true_pred_config={}
        true_pred_config['pred_name'] = 'linear_regression'

        # Save
        files[rep]['data'] = _data_syn
        files[rep]['data_train'] = transformed_train
        files[rep]['data_test'] = transformed_test
        files[rep]['true_dag'] = dag_pd

        files[rep]['target_name'] = target_name
        files[rep]['exposure_names'] = [exposure_name]
        files[rep]['target_idx'] = target_idx
        files[rep]['exposure_idx'] = exposure_idx
        files[rep]['true_mb_target'] = true_mb_target

        files[rep]['true_MB_names'] = true_MB_names
        files[rep]['true_MB_idx'] = true_MB_idx

        files[rep]['true_study_idx'] = true_study_idx
        files[rep]['true_study_names'] = true_study_names

        files[rep]['true_pag_study'] = true_pag_study
        files[rep]['true_mag_study'] = true_mag_study

        files[rep]['true_pred_config'] = true_pred_config

        rep += 1

    with open(files_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)

    with open(params_name, 'wb') as outp:
        pickle.dump(params, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()




