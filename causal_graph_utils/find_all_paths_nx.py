import re
import pandas as pd
import numpy as np
import networkx as nx


def find_all_paths_nx(graph, source_name, target_name, length=5):

    '''
    Find and group all paths between source and target nodes up to length k
    Author : kbiza@csd.uoc.gr
    Args:
        graph(pandas Dataframe): matrix of the causal graph
            graph(i, j) = 2 and graph(j, i) = 3: i-->j
            graph(i, j) = 2 and graph(j, i) = 2: i<->j
            graph(i, j) = 2 and graph(j, i) = 1: io->j
        source_name(str): the name of the source node in the graph Dataframe
        target_name(str): the name of the target node in the graph Dataframe
    Returns:
        paths(dictionary): each key corresponds to a group of paths and contains lists with variable names for each path
            'all':  all identified paths
            'noncausal': only the noncausal paths
            'blocking': only the blocking paths
            'confounding': only the confounding paths
            'potentially': only the potentially directed paths
            'directed':  only the causal paths
    '''


    graph_t = np.transpose(graph)

    G_ones = np.zeros((graph.shape), dtype=int)
    G_ones[np.where(np.logical_and(graph != 0, graph_t != 0))] = 1
    G = nx.from_numpy_array(G_ones, create_using=nx.Graph())

    source_idx = graph.columns.get_loc(source_name)
    target_idx = graph.columns.get_loc(target_name)


    all_paths_nx = []
    all_paths_names = []
    non_causal_paths = []
    directed_paths = []
    potentially_directed_paths = []
    blocking_paths = []
    confounding_paths = []

    for path in sorted(nx.all_simple_paths(G, source_idx, target_idx, cutoff=length)):
        all_paths_nx.append(path)
        all_paths_names.append(graph.columns[path].to_list())

    # examine the type of each path
    for path in all_paths_nx:

        is_directed = True
        is_potentially_directed = True
        is_blocking = False
        is_confounding = False
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            r1 = np.logical_and(graph.iloc[node_i, node_j] == 2, graph.iloc[node_j, node_i] == 3)
            r2 = np.logical_and(graph.iloc[node_i, node_j] == 2, graph.iloc[node_j, node_i] == 1)
            r3 = np.logical_and(graph.iloc[node_i, node_j] == 1, graph.iloc[node_j, node_i] == 1)

            # blocking and confounding paths
            if i < len(path) - 2:
                node_k = path[i+2]
                r4 = np.logical_and(graph.iloc[node_i, node_j] == 2, graph.iloc[node_k, node_j] == 2)
                r5 = np.logical_and(graph.iloc[node_i, node_j] == 3, graph.iloc[node_k, node_j] == 3)

                if r4:
                    is_blocking = True

                if r5:
                    is_confounding = True

            # directed and potentially directed path
            if not r1:
                is_directed = False

            if not r1 and not r2 and not r3:
                is_potentially_directed = False

        if is_directed:
            directed_paths.append(graph.columns[path].to_list())

        elif not is_directed and is_potentially_directed:
            potentially_directed_paths.append(graph.columns[path].to_list())

        else:
            if is_blocking:
                blocking_paths.append(graph.columns[path].to_list())
            elif is_confounding: # if it is blocking it is not confounding
                confounding_paths.append(graph.columns[path].to_list())
            else:
                non_causal_paths.append(graph.columns[path].to_list())


    paths={'all':all_paths_names,
          'noncausal':non_causal_paths,
           'blocking': blocking_paths,
           'confounding': confounding_paths,
           'potentially': potentially_directed_paths,
           'directed': directed_paths
           }

    return paths
