
import numpy as np
import pandas as pd
import networkx as nx
from AutoCD.causal_graph_utils.orientation_rules_cpdag import *
from AutoCD.causal_graph_utils.enforce_stationarity import *

#  Functions to convert CPDAG TO DAG
#  Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr


def FCI_rules_apply(graph, verbose):

    # Applies only rules R1-R3 from FCI.

    flag = True
    while flag:
        flag = False
        graph, flag = R1(graph, flag, verbose)
        graph, flag = R2(graph, flag, verbose)
        graph, flag = R3(graph, flag, verbose)

    return graph

def cpdag_to_dag(cpdag_pd, verbose, n_lags=None):

    '''
    Converts CPDAG to DAG
    Parameters
    ----------
        cpdag_pd (pandas Dataframe): the matrix of the CPDAG
        verbose (bool)
        n_lags (int): the maximum number of previous time lags in case of time-lagged graphs

    Returns
    -------
        dag_pd (pandas Dataframe) : the matrix of the DAG
    '''

    cpdag = cpdag_pd.to_numpy()
    dag = cpdag.copy()

    if isinstance(n_lags, int):
        dag = enforce_stationarity_arrowheads(dag, cpdag_pd, n_lags, verbose)

    # Find circles in o-> edges and turn them into tails
    cpdag_t = np.matrix.transpose(cpdag)

    # Orient the circle component according to Meek's algorithm for chordal graphs.
    # This only works if the graph is chordal.

    # find the circle component
    cpdag_c = np.zeros(cpdag.shape, dtype=int)
    components = np.logical_and(cpdag == 1, cpdag_t == 1)
    cpdag_c[components] = 1

    pag_c_nx = nx.from_numpy_array(cpdag_c)
    ischordal = nx.is_chordal(pag_c_nx)

    if np.count_nonzero(cpdag_c) > 0 and not ischordal:
        sat = False
        print('The circle component is not chordal. Output may not be a correct DAG\n')

    while np.any(dag == 1):
        # pick an edge
        [x_, y_] = np.where(cpdag_c == 1)

        x = x_[0]
        y = y_[0]

        dag[x, y] = 2
        dag[y, x] = 3
        cpdag_c[x, y] = 0
        cpdag_c[y, x] = 0

        if verbose:
            print('R0 Orienting %d-->%d\n' % (y, x))

        dag = FCI_rules_apply(dag, verbose)

    dag_pd = pd.DataFrame(dag, columns=cpdag_pd.columns, index=cpdag_pd.index)

    return dag_pd