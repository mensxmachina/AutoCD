import numpy as np
import pandas as pd
import networkx as nx
from AutoCD.causal_graph_utils.enforce_stationarity import *
from AutoCD.causal_graph_utils.orientation_rules_cpdag import *
from AutoCD.causal_graph_utils.get_unshielded_triples import *


#  Functions to convert DAG to CPDAG
#  Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr

def FCI_rules_dag(G, dag, verbose):

    flagcount = 0
    unshielded_triples = get_unshielded_triples(G)

    G, dnc = R0(G, unshielded_triples, dag, verbose)

    flag = True

    while flag:
        flag = False
        G, flag = R1(G, flag, verbose)
        G, flag = R2(G, flag, verbose)
        G, flag = R3(G, flag, verbose)
        flagcount = flagcount + int(flag)

    return G, dnc, flagcount


def dag_to_cpdag(dag_pd, verbose, n_lags=None):

    '''
    Converts DAG to CPDAG
    Parameters
    ----------
        dag_pd (pandas Dataframe) : the matrix of the DAG
        verbose (bool)
        n_lags(int or None): the maximum number of previous time lags
                             if int, the dag_pd must be a time-lagged graph
    Returns
    -------
        cpdag_pd (pandas Dataframe): the matrix of the CPDAG
    '''

    dag = dag_pd.to_numpy()
    cpdag = dag.copy()
    cpdag[cpdag != 0] = 1

    if isinstance(n_lags, int):
        cpdag = enforce_stationarity_arrowheads(cpdag, dag_pd, n_lags, verbose)

    cpdag, dnc, flagcount = FCI_rules_dag(cpdag, dag, verbose)

    if isinstance(n_lags, int):
        cpdag = enforce_stationarity_tails_and_orientation(cpdag, dag_pd, n_lags, verbose)


    cpdag_pd = pd.DataFrame(cpdag, columns=dag_pd.columns, index=dag_pd.index)

    return cpdag_pd
