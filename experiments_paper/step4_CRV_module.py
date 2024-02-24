import re
import pandas as pd
import numpy as np
import jpype
from jpype import *
from jpype.types import *
import jpype.imports
import pickle

from sklearn.utils import resample
from sklearn import metrics
from AutoCD.causal_discovery.functions_causal_discovery import *
from AutoCD.metrics_evaluation.edge_metrics_bootstraps import *


# ----------------------------------------------------------------------------------------------
# Experiment: Apply the CR module on the estimated causal graph to compute  the edge confidences
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------

def main():

    jpype.startJVM("-ea", classpath=['../jar_files/*'], convertStrings=False)

    # file names
    path = './files_results/'
    id = '20n_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb_cd.pkl'
    output_name = path + id + 'files_mb_cd_boot.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    # parameters for bootstrapping
    window_resampling = 2 * params['n_lags']
    B = 50

    # experiment
    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)

        true_dag = files[rep]['true_dag']
        all_vars = true_dag.columns
        true_pag_study = files[rep]['true_pag_study']
        opt_pag_study = files[rep]['opt_pag_study']
        dataObj = files[rep]['dataObj']
        opt_causal_config = files[rep]['opt_causal_config']

        # Create bootstrapped samples and apply the selected causal configuration
        samples_for_boostrap = timeseries_to_timelagged(dataObj.samples, window_resampling, window=True)
        bootstrapped_samples_all = []
        bootstrapped_matrix_mec = []

        b = 0
        while b < B:
            bootstrapped_samples_ = resample(samples_for_boostrap,
                                             n_samples=samples_for_boostrap.shape[0], replace=True)

            bootstrapped_samples = timelagged_to_timeseries(bootstrapped_samples_, window_resampling)

            library_results, boost_mec_graph, boost_graph = \
                causal_discovery(opt_causal_config, bootstrapped_samples, dataObj.data_type_info, dataObj.is_time_series)

            if isinstance(boost_mec_graph, pd.DataFrame):
                bootstrapped_samples_all.append(bootstrapped_samples)

                boost_pag_study = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                              columns=all_vars, index=all_vars)
                boost_pag_study.loc[boost_mec_graph.index, boost_mec_graph.columns] = (
                    boost_mec_graph.loc)[boost_mec_graph.index, boost_mec_graph.columns]

                bootstrapped_matrix_mec.append(boost_pag_study)

                b += 1


        # Compute edge consistency and discovery frequency on bootstrapped graphs
        edge_consistency, matrix_consistency = edge_metrics_on_bootstraps(
            best_mec_matrix=opt_pag_study, bootstrapped_mec_matrix=bootstrapped_matrix_mec)

        edge_consistency_wrt_true, matrix_consistency_wrt_true = edge_metrics_on_bootstraps(
            best_mec_matrix=true_pag_study, bootstrapped_mec_matrix=[opt_pag_study])

        # Compute AUC for edge consistency frequency
        nnz_true = true_pag_study > 0
        nnz_true_np = nnz_true.to_numpy()
        matrix_consistency_np = matrix_consistency.to_numpy()

        consistency_vector= matrix_consistency_np[np.triu_indices(matrix_consistency_np.shape[1])]
        true_vector = nnz_true_np[np.triu_indices(nnz_true_np.shape[1])]
        fpr, tpr, thresholds = metrics.roc_curve(true_vector, consistency_vector, pos_label=1)
        auc_ = metrics.auc(fpr, tpr)


        # Save
        files[rep]['bootstrapped_samples'] = bootstrapped_samples_all
        files[rep]['bootstrapped_mec'] = bootstrapped_matrix_mec
        files[rep]['edge_consistency'] = edge_consistency
        files[rep]['matrix_consistency'] = matrix_consistency
        files[rep]['edge_consistency_true'] = edge_consistency_wrt_true
        files[rep]['matrix_consistency_true'] = matrix_consistency_wrt_true
        files[rep]['boot_auc'] = auc_


    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()