import networkx as nx
import pandas as pd
import numpy as np


def is_dag(dag_pd):

    '''
    Checks if the input graph is a DAG
    Parameters
    ----------
        dag_pd(pandas Dataframe): the matrix of the graph

    Returns
    -------
        is_dag(bool)
    '''

    dag = dag_pd.to_numpy()
    dag_t = np.transpose(dag)

    G_ones = np.zeros((dag.shape), dtype=int)
    G_ones[np.where(np.logical_and(dag == 2, dag_t == 3))] = 1
    G = nx.from_numpy_array(G_ones, create_using=nx.DiGraph())
    is_acyclic_ = nx.is_directed_acyclic_graph(G)

    undirected = np.where(dag == 1)[0]
    arrows = np.where(dag == 2)[0]
    tails = np.where(dag_t == 3)[0]
    proper_directed_edges = set(arrows) == set(tails)

    if proper_directed_edges and is_acyclic_ and len(undirected) == 0:
        is_dag = True
    else:
        is_dag = False

    return is_dag

