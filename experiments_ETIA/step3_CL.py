
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
# Experiment: Apply the CL module on the reduced dataset to estimate the causal graph
# Author: kbiza@csd.uoc.gr
# ---------------------------------------------------------------------------------------------------


def main():

    jpype.startJVM("-ea", classpath=['../jar_files/*'], convertStrings=False)

    # file names
    path = './files_results/'
    id = '100n_2500s_3ad_6md_1exp_1rep_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb.pkl'
    output_name = path + id + 'files_mb_cd.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        true_dag = files[rep]['true_dag']
        dataObj = files[rep]['dataObj']
        true_pag_study = files[rep]['true_pag_study']
        true_study_idx = files[rep]['true_study_idx']
        est_study_idx = files[rep]['est_study_idx']

        # Create causal configurations
        causal_configs = CausalConfigurator().create_causal_configs(dataObj, False)

        # Run OCT tuning method
        library_results, est_pag_study, est_mag_study, opt_causal_config = OCT_parallel(dataObj, 2).select(causal_configs)

        # Graph matrices over all vars
        all_vars = true_dag.columns
        opt_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        opt_pag_all.loc[est_pag_study.index, est_pag_study.columns] = \
            (est_pag_study.loc)[est_pag_study.index, est_pag_study.columns]

        true_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        true_pag_all.loc[true_pag_study.index, true_pag_study.columns] = \
            (true_pag_study.loc)[true_pag_study.index, true_pag_study.columns]

        # Graph matrices over the union of true and est variables
        union_study_idx = list(set(true_study_idx + est_study_idx))
        union_study_idx.sort()
        union_study_names = true_dag.columns[union_study_idx]

        true_pag_union = true_pag_all.loc[union_study_names, union_study_names]
        opt_pag_union = opt_pag_all.loc[union_study_names, union_study_names]

        # Compute SHD
        shd_opt = shd_mag_pag(true_pag_union.to_numpy(), opt_pag_union.to_numpy())

        # Compute adjacency precision and recall
        adj_prec, adj_rec = adjacency_precision_recall(true_pag_union, opt_pag_union)

        # Evaluate OCT method
        shds_all = np.zeros((len(causal_configs), 1))
        est_pag_union_all = []
        for c, config in enumerate(causal_configs):
            _, mec_graph_i, _ = \
                causal_discovery(config, dataObj.samples, dataObj.data_type_info, params['is_time_series'])

            est_pag_i_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                           columns=all_vars, index=all_vars)

            est_pag_i_all.loc[mec_graph_i.index, mec_graph_i.columns] = (
                mec_graph_i.loc)[mec_graph_i.index, mec_graph_i.columns]

            est_pag_i_union = est_pag_i_all.loc[union_study_names, union_study_names]

            shds_all[c, 0] = shd_mag_pag(true_pag_union.to_numpy(), est_pag_i_union.to_numpy())
            est_pag_union_all.append(est_pag_i_union)

        # Save
        files[rep]['est_pag_study'] = est_pag_study
        files[rep]['est_mag_study'] = est_mag_study
        files[rep]['opt_causal_config'] = opt_causal_config

        files[rep]['union_study_names'] = union_study_names
        files[rep]['opt_pag_union'] = opt_pag_union
        files[rep]['true_pag_union'] = true_pag_union
        files[rep]['est_pag_union_all'] = est_pag_union_all

        files[rep]['shd'] = shd_opt
        files[rep]['adj_prec'] = adj_prec
        files[rep]['adj_rec'] = adj_rec
        files[rep]['delta_shd'] = shd_opt - np.min(shds_all)
        files[rep]['shds_all'] = shds_all

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