import numpy as np
import networkx as nx
import pandas as pd
from AutoCD.causal_graph_utils.orientation_rules import *
from AutoCD.causal_graph_utils.enforce_stationarity import *


#  Functions to convert PAG TO MAG
#  Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr


def FCI_rules_apply_(graph, verbose):
    # Applies only rules R1-R3 from FCI.
    flag = True
    while flag:
        flag = False
        graph, flag = R1(graph, flag, verbose)
        graph, flag = R2_(graph, flag, verbose)
        graph, flag = R3(graph, flag, verbose)

    return graph


def pag_to_mag(pag_pd, verbose, n_lags=None):

    '''
    Converts PAG to MAG
    Parameters
    ----------
        pag_pd (pandas Dataframe): the matrix of the PAG
        verbose (bool)
        n_lags (int): the maximum number of previous time lags in case of time-lagged graphs

    Returns
    -------
        mag_pd (pandas Dataframe) : the matrix of the MAG
    '''

    pag_np = pag_pd.to_numpy()
    mag = pag_np.copy()

    if isinstance(n_lags, int):
        mag = enforce_stationarity_arrowheads(mag, pag_pd, n_lags, verbose)

    # Find circles in o-> edges and turn them into tails
    pag_t = np.matrix.transpose(pag_np)
    circles = np.logical_and(pag_np == 1, pag_t == 2)
    mag[circles] = 3

    # Orient the circle component according to Meek's algorithm for chordal
    # graphs. This only works if the graph is chordal.

    # find the circle component
    pag_c = np.zeros(pag_np.shape, dtype=int)
    components = np.logical_and(pag_np == 1, pag_t == 1)
    pag_c[components] = 1

    pag_c_nx = nx.from_numpy_array(pag_c)
    ischordal = nx.is_chordal(pag_c_nx)

    if np.count_nonzero(pag_c) > 0 and not ischordal:
        sat = False
        print('The circle component is not chordal. Output may not be a correct MAG\n')

    while np.any(mag == 1):
        # pick an edge
        [x_, y_] = np.where(pag_c == 1)

        x = x_[0]
        y = y_[0]

        mag[x, y] = 2
        mag[y, x] = 3
        pag_c[x, y] = 0
        pag_c[y, x] = 0

        if verbose:
            print('Orienting %d->%d\n' % (y, x))

        mag = FCI_rules_apply_(mag, verbose)

    mag_pd=pd.DataFrame(mag, columns=pag_pd.columns, index=pag_pd.index)

    return mag_pd
