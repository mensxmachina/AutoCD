import re
import pandas as pd
import numpy as np
import jpype
from jpype import *
from jpype.types import *
import jpype.imports
import pickle

from sklearn.utils import resample
from AutoCD.causal_discovery.functions_causal_discovery import *

# ----------------------------------------------------------------------------------------------
# Experiment: Apply the CRV module on the estimated causal graph to create bootstrapped graphs
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------


def main():

    jpype.startJVM("-ea", classpath=['../jar_files/*'], convertStrings=False)

    # file names
    path = './files_results/'
    id = '100n_2500s_3ad_6md_1exp_1rep_'
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb_cd_adjset_cmse_minimal.pkl'
    output_name = path + id + 'files_mb_cd_adjset_cmse_minimal_boot.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    B = 100
    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)

        all_vars = files[rep]['true_dag'].columns
        union_study_names = files[rep]['opt_pag_union'].columns

        dataObj = files[rep]['dataObj']
        opt_causal_config = files[rep]['opt_causal_config']

        # Create bootstrapped samples and apply the selected causal configuration
        samples_for_boostrap = dataObj.samples.copy()
        bootstrapped_samples_all = []
        bootstrapped_matrix_mec = []

        b = 0
        while b < B:
            bootstrapped_samples = resample(samples_for_boostrap,
                                             n_samples=samples_for_boostrap.shape[0], replace=True)

            library_results, boost_mec_graph, boost_graph = \
                causal_discovery(opt_causal_config, bootstrapped_samples, dataObj.data_type_info, dataObj.is_time_series)

            if isinstance(boost_mec_graph, pd.DataFrame):
                bootstrapped_samples_all.append(bootstrapped_samples)

                boost_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                              columns=all_vars, index=all_vars)
                boost_pag_all.loc[boost_mec_graph.index, boost_mec_graph.columns] = (
                    boost_mec_graph.loc)[boost_mec_graph.index, boost_mec_graph.columns]

                boost_pag_union = boost_pag_all.loc[union_study_names, union_study_names]

                bootstrapped_matrix_mec.append(boost_pag_union)

                b += 1

        # Save
        files[rep]['bootstrapped_samples'] = bootstrapped_samples_all
        files[rep]['bootstrapped_mec'] = bootstrapped_matrix_mec

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()