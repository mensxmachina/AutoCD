import re
import pandas as pd
import numpy as np
import pickle

from AutoCD.adjustment.function_find_adjset_daggity import *

# ----------------------------------------------------------------------------------------------
# Experiment: Apply the CRV module on the estimated causal graph to find the Adjustment Sets
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------


def main():

    # file names
    path = './files_results/'
    id = '100n_2500s_3ad_6md_1exp_1rep_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb_cd.pkl'
    output_name = path + id + 'files_mb_cd_adjset.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)

        true_dag = files[rep]['true_dag']
        true_mag_study = files[rep]['true_mag_study']
        est_pag_study = files[rep]['est_pag_study']
        target_name = files[rep]['target_name']
        exposure_names = files[rep]['exposure_names']

        # Adjustment set in true DAG and MAG
        Z_true_can_dag, Z_true_min_dag = find_adjset(true_dag, 'dag', [target_name], exposure_names)
        Z_true_can_mag, Z_true_min_mag = find_adjset(true_mag_study, 'mag', [target_name], exposure_names)

        # Adjustment set
        Z_est_can_pag, Z_est_min_pag = find_adjset(est_pag_study, 'pag', [target_name], exposure_names)

        # Save
        files[rep]['Z_true_can_dag'] = Z_true_can_dag
        files[rep]['Z_true_min_dag'] = Z_true_min_dag
        files[rep]['Z_true_can_mag'] = Z_true_can_mag
        files[rep]['Z_true_min_mag'] = Z_true_min_mag

        files[rep]['Z_est_can_pag'] = Z_est_can_pag
        files[rep]['Z_est_min_pag'] = Z_est_min_pag

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()