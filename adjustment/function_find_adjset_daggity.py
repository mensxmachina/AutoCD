import numpy as np
import pandas as pd
from AutoCD.adjustment.adjset_R import *


def find_adjset(graph_pd, graph_type, target_name, exposure_names):

    '''
    Run the dagitty R package to identify the adjustment set of X and Y
    Author: kbiza@csd.uoc.gr
    Args:
        graph_pd(pandas Dataframe): the graph as adjacency matrix
        graph_type(str): the type of the graph : {'dag', 'cpdag', 'mag', 'pag'}
        target_name: list of one variable name
        exposure_names:  list of one or more variable names

    Returns:
        adj_set_can(list): the variable names of the canonical adj. set (if exists)
        adj_set_min(list):: the variable names of the minimal adj. set (if exists)
    '''

    graph_np = graph_pd.to_numpy()
    if graph_type in ['dag', 'cpdag']:
        pcalg_graph = np.zeros(graph_np.shape, dtype=int)
        pcalg_graph[graph_np == 1] = 1
        pcalg_graph[graph_np == 2] = 1
        pcalg_graph_t = np.transpose(pcalg_graph)
        pcalg_graph_pd = pd.DataFrame(pcalg_graph_t, index=graph_pd.index, columns=graph_pd.columns)
    else:
        pcalg_graph_pd = graph_pd.copy()

    exposure_names_ = [sub.replace(':', '.') for sub in exposure_names]
    canonical_dg, minimal_dg = adjset_dagitty(pcalg_graph_pd, graph_type, exposure_names_, target_name)

    if isinstance(canonical_dg, list):
        adj_set_can_ = canonical_dg[0]
        adj_set_can = [sub.replace('.', ':') for sub in adj_set_can_]
    else:
        adj_set_can = None

    if isinstance(minimal_dg, list):
        adj_set_min_ = minimal_dg[0]
        adj_set_min = [sub.replace('.', ':') for sub in adj_set_min_]
    else:
        adj_set_min = None

    return adj_set_can, adj_set_min