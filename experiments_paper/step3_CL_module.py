
import json
import pickle
import time
import pandas as pd
import numpy as np
import jpype
from jpype import *
from jpype.types import *
import jpype.imports

from AutoCD.causal_discovery.select_with_OCT_parallel import *
from AutoCD.causal_discovery.class_causal_config import *
from AutoCD.metrics_evaluation.adjacency_precision_recall import *
from AutoCD.metrics_evaluation.shd_mag_pag import *


# ---------------------------------------------------------------------------------------------------
# Experiment: Apply the CL module on the reduced simulated temporal data to estimate the causal graph
# Author: kbiza@csd.uoc.gr
# ---------------------------------------------------------------------------------------------------


def main():

    jpype.startJVM("-ea", classpath=['../jar_files/*'], convertStrings=False)

    # file names
    path = './files_results/'
    id = '20n_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb.pkl'
    output_name = path + id + 'files_mb_cd.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    # experiment
    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        true_dag = files[rep]['true_dag']
        all_vars = true_dag.columns
        dataObj = files[rep]['dataObj']
        true_pag_study = files[rep]['true_pag_study']

        # Create causal configurations
        causal_configs = CausalConfigurator().create_causal_configs(dataObj, False)

        # Run OCT tuning method
        library_results, opt_mec_graph_pd, opt_graph_pd, opt_causal_config = OCT_parallel(dataObj, 2).select(causal_configs)

        opt_pag_study = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        opt_pag_study.loc[opt_mec_graph_pd.index, opt_mec_graph_pd.columns] = \
            (opt_mec_graph_pd.loc)[opt_mec_graph_pd.index, opt_mec_graph_pd.columns]


        # Compute SHD
        shd_opt = shd_mag_pag(true_pag_study.to_numpy(), opt_pag_study.to_numpy())

        # Compute adjacency precision and recall
        adj_prec, adj_rec = adjacency_precision_recall(true_pag_study, opt_pag_study)

        # Evaluate OCT method
        shds_all = np.zeros((len(causal_configs), 1))
        est_pag_study_all = []
        for c, config in enumerate(causal_configs):
            _, mec_graph_i, _ = \
                causal_discovery(config, dataObj.samples, dataObj.data_type_info, params['is_time_series'])

            est_pag_i_study = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                           columns=all_vars, index=all_vars)

            est_pag_i_study.loc[mec_graph_i.index, mec_graph_i.columns] = (
                mec_graph_i.loc)[mec_graph_i.index, mec_graph_i.columns]

            shds_all[c, 0] = shd_mag_pag(true_pag_study.to_numpy(), est_pag_i_study.to_numpy())
            est_pag_study_all.append(est_pag_i_study)


        # Save
        files[rep]['opt_mec'] = opt_mec_graph_pd
        files[rep]['opt_graph'] = opt_graph_pd
        files[rep]['opt_causal_config'] = opt_causal_config
        files[rep]['true_pag_study'] = true_pag_study
        files[rep]['opt_pag_study'] = opt_pag_study
        files[rep]['shd'] = shd_opt
        files[rep]['adj_prec'] = adj_prec
        files[rep]['adj_rec'] = adj_rec
        files[rep]['delta_shd'] = shd_opt - np.min(shds_all)
        files[rep]['shds_all'] = shds_all
        files[rep]['est_pag_study_all'] = est_pag_study_all
        files[rep]['cd_exec_time'] = time.time() - t0
        files[rep]['causal_configs'] = causal_configs # similar for all repetitions

    # Print results
    adj_prec_ = [d.get('adj_prec') for d in files]
    print('mean Adj.Precision: %0.2f' %(np.mean(adj_prec_)), 'SE:%0.2f' %(np.std(adj_prec_) / np.sqrt(len(adj_prec_))))

    adj_rec_ = [d.get('adj_rec') for d in files]
    print('mean Adj. Recall: %0.2f' %(np.mean(adj_rec_)), 'SE:%0.2f' %(np.std(adj_rec_) / np.sqrt(len(adj_rec_))))

    delta_shd_ = [d.get('delta_shd') for d in files]
    print('mean DSHD:%0.2f' %(np.mean(delta_shd_)), 'SE:%0.2f' %(np.std(delta_shd_) / np.sqrt(len(delta_shd_))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()