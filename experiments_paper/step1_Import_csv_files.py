
import jpype
import numpy as np
from jpype import *
from jpype.types import *
import jpype.imports
import json
import pickle

from AutoCD.data_utils.function_simulate_tigramite import *
from AutoCD.data_utils.functions_data import *
from AutoCD.causal_graph_utils.markov_boundary import *
from AutoCD.causal_graph_utils.create_sub_mag_pag import *


# ------------------------------------------------------------------------------
# Experiment: Import csv files and create the necessary files for the experiment
#              Change the number of nodes (params['n_nodes']= {20, 50, 100})
# Author: kbiza@csd.uoc.gr
# ------------------------------------------------------------------------------

def main():

    # Simulation parameters
    params = {}
    params['n_nodes'] = 100  # change only the number of nodes {20, 50, 100}

    params['n_rep'] = 10
    params['n_lags'] = 2
    params['n_samples'] = 2500
    params['n_train'] = 500
    params['is_time_series'] = True

    csv_path = './csv_files/'+str(params['n_nodes'])+'nodes/'
    path = './files_results/'
    id_file = (str(params['n_nodes']) + 'n_')
    files_name = path + id_file + 'files.pkl'
    params_name = path + id_file + 'params.pkl'

    files = [{} for _ in range(params['n_rep'])]

    for rep in range(params['n_rep']):

        dag_pd = pd.read_csv(csv_path+'true_dag_'+str(params['n_nodes'])+'_'+str(rep)+'.csv', index_col=0)
        _data_syn =pd.read_csv(csv_path+'data_' + str(params['n_nodes']) + '_' + str(rep) + '.csv', index_col=0)
        target_name_pd = pd.read_csv(csv_path+'target_name_' + str(params['n_nodes']) + '_' + str(rep) + '.csv', index_col=0)
        target_name = target_name_pd['target_name'][0]
        target_idx = dag_pd.columns.get_loc(target_name)

        # True Mb
        true_dag = dag_pd.copy()
        true_mb_idx = markov_boundary(target_idx, true_dag)
        true_mb_nameslag = true_dag.columns[true_mb_idx]
        true_mb_names = names_from_lag(true_mb_nameslag)

        # True PAG using true MB
        all_vars = true_dag.columns
        study_vars_true = np.concatenate((true_mb_names, [target_name]))
        study_vars_true = list(set(study_vars_true))
        _, true_pag_ = create_sub_mag_pag(true_dag, study_vars_true, params['n_lags'])

        true_pag_study = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        true_pag_study.loc[true_pag_.index, true_pag_.columns] = true_pag_.loc[true_pag_.index, true_pag_.columns]


        # Transform data
        data_train = _data_syn.iloc[0:params['n_train'],:].copy()
        data_test = _data_syn.iloc[params['n_train']:-1, :].copy()

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
        files[rep]['target_name'] = target_name
        files[rep]['true_dag'] = dag_pd
        files[rep]['true_mb_names'] = true_mb_names
        files[rep]['true_mb_nameslag'] = true_mb_nameslag
        files[rep]['true_pag_study'] = true_pag_study
        files[rep]['true_pred_config'] = true_pred_config

        rep+=1

    with open(files_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)

    with open(params_name, 'wb') as outp:
        pickle.dump(params, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()




