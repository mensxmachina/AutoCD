
import numpy as np
import networkx as nx

def find_ancestors_nx(graph, node=None):

    '''
    A is an ancestor of B if graph(i,j)=2 and graph(j,i)=3
    for every edge i-->j in the path from A to B
    Author : kbiza@csd.uoc.gr
    Args:
        graph(numpy array): matrix of the causal graph
            graph(i, j) = 2 and graph(j, i) = 3: i-->j
            graph(i, j) = 2 and graph(j, i) = 2: i<->j
            graph(i, j) = 2 and graph(j, i) = 1: io->j
        node(int): the node of interest to find its ancestors
            if None it returns the ancestors of all nodes
    Returns:
        is_ancestor:
            (list) : if a node is given it returns the indexes of its ancestors
            (numpy array): if no node is given it finds the ancestors of all nodes
                            and returns logical matrix

    Note: the node under study is not in the set of its ancestors
    '''

    n_nodes = graph.shape[1]
    graph_t = np.transpose(graph)

    G_ones = np.zeros((graph.shape), dtype=int)
    G_ones[np.where(np.logical_and(graph == 2, graph_t==3))] = 1
    G = nx.from_numpy_array(G_ones, create_using=nx.MultiDiGraph())
    TC = nx.transitive_closure_dag(G)

    if isinstance(node, int):
        return list(TC.predecessors(node))
    else:
        # find ancestors for each node
        is_ancestor = np.zeros((n_nodes, n_nodes), dtype=bool)
        for node in range(n_nodes):
            cur_ancestors = list(TC.predecessors(node))
            is_ancestor[cur_ancestors, node] = True
            # is_ancestor[node, node] = True

        return is_ancestor