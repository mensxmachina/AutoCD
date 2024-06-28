import re
import pandas as pd
import numpy as np
import pickle

from AutoCD.visualization.plot_in_cytoscape import *
from AutoCD.causal_graph_utils.confidence_causal_findings import *
from AutoCD.causal_graph_utils.find_all_paths_nx import *

# ----------------------------------------------------------------------------------------------
# Experiment: Apply the CRV module to visualize the estimated PAG and subgraphs of interest
# Launch Cytoscape before running the script
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------

path = './files_results/'
id = '100n_2500s_3ad_6md_1exp_1rep_'    # change the file name if needed
path_len = 5                            # change the maximum length of the extracted paths


# load data and simulation parameters
input_name = path + id + 'files_mb_cd_adjset_cmse_minimal_boot_evalConf.pkl'
with open(input_name, 'rb') as inp:
    files = pickle.load(inp)


for rep in range(len(files)):

    target_name = files[rep]['target_name']
    exposure_names = files[rep]['exposure_names']
    opt_pag_union = files[rep]['opt_pag_union']
    weight_data_pd_all = files[rep]['weight_data_pd_all']
    Z_est_min_pag = files[rep]['Z_est_min_pag']

    # Visualize the estimated graph
    net_suid = visualize_graph(opt_pag_union,
                    'estimated_PAG', 'ETIA','pag',
                      target_name = target_name,
                      exposure = exposure_names,
                      adj_set = Z_est_min_pag,
                      edge_info = weight_data_pd_all, edge_weights='edge_consistency', show_weights=False)

    # Find paths up to length k on the estimated graph between exposure and target
    paths = find_all_paths_nx(opt_pag_union, exposure_names[0], target_name, length=path_len)
    path_consistency, path_discovery = paths_metrics(opt_pag_union, files[rep]['bootstrapped_mec'], paths)

    # Visualize paths as subgraphs
    visualize_subpgraphs(paths, net_suid, path_confidence=path_consistency, separate_paths=True)


