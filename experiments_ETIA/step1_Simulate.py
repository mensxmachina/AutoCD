
import jpype
import numpy as np
from jpype import *
from jpype.types import *
import jpype.imports
import pickle

from AutoCD.data_utils.function_simulate_tigramite import *
from AutoCD.data_utils.functions_data import *
from AutoCD.causal_graph_utils.create_sub_mag_pag import *
from AutoCD.causal_graph_utils.markov_boundary import *
from AutoCD.causal_graph_utils.cpdag_to_dag import *

# ----------------------------------------------------------------------
# Experiment: Simulate cross-sectional data using the Tetrad project
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------


def main():

    jpype.startJVM("-ea", classpath=['../jar_files/*'], convertStrings=False)

    # Simulation parameters
    params={}
    params['n_rep'] = 1                 # number of repetitions
    params['n_nodes'] = 100             # number of variables
    params['n_samples'] = 2500          # total number of samples
    params['n_test'] = 500              # number of samples to hold out
    params['avg_degree'] = 3            # average node degree
    params['max_degree'] = 6            # maximum node degree
    params['simulation_type'] = 'GSEM'  # GSEM, BayesNet, LeeHastie, CondGauss,TimeSeries

    params['n_exposures'] = 1           # number of exposure variables
    params['len_path_exp'] = 1          # minimum length of the directed path from exposure to target
    params['len_parent_target'] = 1     # minimum number of parents for the target
    params['is_time_series'] = False
    params['n_lags'] = None


    # File names
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


    # Simulation
    files = [{} for _ in range(params['n_rep'])]
    rep = 0
    while rep < params['n_rep']:

        _, _, _, dag_pd, cpdag_pd, _data_syn = \
            simulate_data_tetrad(params['n_nodes'],
                                 params['n_samples'],
                                 params['avg_degree'],
                                 params['max_degree'],
                                 params['simulation_type'])

        # Random target and exposures
        # Target has at least one parent
        flag_target = True
        while flag_target:
            print('search target')
            target_idx = np.random.randint(0, params['n_nodes'])
            if np.count_nonzero(dag_pd.iloc[:, target_idx] == 2) > params['len_parent_target']:
                flag_target = False

        # Each exposure has at least one directed path towards target
        flag_exposure = True
        n_exp = 0
        exposure_idx = []
        while flag_exposure:
            print('search for exposure')
            idx_ = np.random.randint(low=0, high=params['n_nodes'])
            if idx_ == target_idx or idx_ in exposure_idx:
                continue
            path = one_directed_path(dag_pd.to_numpy(), idx_, target_idx)
            if isinstance(path, list) and len(path) > params['len_path_exp']:
                exposure_idx = exposure_idx + [idx_]
                n_exp += 1
            if n_exp == params['n_exposures']:
                flag_exposure = False

        target_name = 'V'+str(target_idx+1)
        exposure_names = ['V'+str(e_idx+1) for e_idx in exposure_idx]

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
        files[rep]['exposure_names'] = exposure_names
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
