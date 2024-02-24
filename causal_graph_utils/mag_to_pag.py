
import pandas as pd
from AutoCD.causal_graph_utils.orientation_rules import *
from AutoCD.causal_graph_utils.enforce_stationarity import *
from AutoCD.causal_graph_utils.get_unshielded_triples import *


# Functions to convert MAG to PAG
# Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr


def FCI_rules_mag(G, mag, verbose):

    '''
    Applies the FCI rules on the given graph
    Parameters
    ----------
        G(numpy matrix): the matrix of the graph
        mag(numpy matrix) :  the matrix of the mag
        verbose (bool)

    Returns
    ---------
        G(numpy matrix): the matrix of the graph
        dnc (dictionary)
        flagcount (int)
    '''

    flagcount = 0
    unshielded_triples = get_unshielded_triples(G)

    G, dnc = R0(G, unshielded_triples, mag, verbose)

    flag = True

    while flag:
        flag = False
        G, flag = R1(G, flag, verbose)
        G, flag = R2_(G, flag, verbose)
        G, flag = R3(G, flag, verbose)
        G, flag = R4(G, mag, flag, verbose)
        flagcount = flagcount + int(flag)

    flag = True
    while flag:
        flag = False
        G, flag = R8(G, flag, verbose)
        G, flag = R9_R10(G, dnc, flag, verbose)
        flagcount = flagcount + int(flag)

    return G, dnc, flagcount


def mag_to_pag(mag_pd, verbose, n_lags=None):

    '''
    Converts MAG to PAG
    Parameters
    ----------
        mag_pd (pandas Dataframe): the matrix of the MAG
        verbose (bool)
        n_lags (int) : the maximum number of previous time lags in case of time-lagged graphs

    Returns
    -------
        pag_pd (pandas Dataframe) : the matrix of the PAG
    '''

    mag = mag_pd.to_numpy()

    pag = mag.copy()
    pag[pag != 0] = 1

    pag, dnc, flagcount = FCI_rules_mag(pag, mag, verbose)

    if isinstance(n_lags, int):
        pag = enforce_stationarity_arrowheads(pag, mag_pd, n_lags, verbose)
        pag = enforce_stationarity_tails_and_orientation(pag, mag_pd, n_lags, verbose)

        pag, dnc, flagcount = FCI_rules_mag(pag, mag, verbose)

    pag_pd = pd.DataFrame(pag, columns=mag_pd.columns, index=mag_pd.index)

    return pag_pd