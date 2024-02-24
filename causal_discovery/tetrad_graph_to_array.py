import re
import pandas as pd
import numpy as np
import jpype
from jpype import *
from jpype.types import *
import jpype.imports

def _tetrad_var_name_to_index(tetrad_var_name, n_lag_nodes=None):

    """
    Finds the matrix index of the given variable's name

    Parameters
    ----------
        tetrad_var_name (str) : variable's name in Tetrad
        n_lag_nodes(int or None) : the number of nodes in a lag in case of time series data

    Returns
    ----------
        var_index(int) : the index in the matrix

    """

    if ':' in tetrad_var_name:
        match = re.search('X(\d+):(\d+)', tetrad_var_name)
        pos = match.group(1)
        lag = match.group(2)
        var_index = int(pos) + (int(lag) * n_lag_nodes)
    else:
        match = re.search('X(\d+)', tetrad_var_name)
        pos = match.group(1)
        var_index = int(pos)

    return var_index


def tetrad_graph_to_array(tetrad_graph_, n_lags=None):

    """
    Covert tetrad graph to numpy array
        Parameters
        ----------
        tetrad_graph_ (Tetrad object) : graph from tetrad (it can be also a time-lagged graph)
        n_lags(int or None) : the number of previous time lags in case of time series

        Returns
        -------
        matrix_pd(pandas Dataframe): matrix of size N*N where N is the number of nodes in tetrad_graph
            matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
            matrix(i, j) = 1 and matrix(j, i) = 1: io-oj   in PAGs
            matrix(i, j) = 2 and matrix(j, i) = 2: i<->j   in MAGs and PAGs
            matrix(i, j) = 3 and matrix(j, i) = 3: i---j   in PDAGs
            matrix(i, j) = 2 and matrix(j, i) = 1: io->j
    """

    n_nodes_ = tetrad_graph_.getNumNodes()
    edges = tetrad_graph_.getEdges()
    edgesIterator = edges.iterator()

    matrix = np.zeros(shape = (n_nodes_, n_nodes_), dtype = int)

    while edgesIterator.hasNext():
        curEdge = edgesIterator.next()

        Nodei = str(curEdge.getNode1().toString())
        Nodej = str(curEdge.getNode2().toString())

        iToj = str(curEdge.getEndpoint2().toString())
        jToi = str(curEdge.getEndpoint1().toString())

        if n_lags:
            i = _tetrad_var_name_to_index(Nodei, int(n_nodes_ / (n_lags + 1)))
            j = _tetrad_var_name_to_index(Nodej, int(n_nodes_ / (n_lags + 1)))
        else:
            i = _tetrad_var_name_to_index(Nodei)
            j = _tetrad_var_name_to_index(Nodej)

        i = i - 1  # python indexing
        j = j - 1  # python indexing


        if iToj == 'Circle' or iToj == 'CIRCLE':
            matrix[i, j] = 1
        elif iToj == 'Arrow' or iToj == 'ARROW':
            matrix[i, j] = 2
        elif iToj == 'Tail' or iToj == 'TAIL':
            matrix[i, j] = 3

        if jToi == 'Circle' or jToi == 'CIRCLE':
            matrix[j, i] = 1
        elif jToi == 'Arrow' or jToi == 'ARROW':
            matrix[j, i] = 2
        elif jToi == 'Tail' or jToi == 'TAIL':
            matrix[j, i] = 3


    # tail - tail corresponds to o-o
    matrix_t = np.transpose(matrix)
    tail_tail = np.logical_and(matrix == 3, matrix_t == 3)
    matrix[tail_tail] = 1

    return matrix
