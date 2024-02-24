import numpy as np
import pandas as pd

def adjacency_precision_recall(true_G_pd, est_G_pd):

    '''
    Computes adjacency precision and recall
    Parameters
    ----------
        true_G_pd(pandas Dataframe): the true graph
        est_G_pd(pandas Dataframe): the estimated graph

    Returns
    -------
        adj_prec(float) : the adjacency precision
        adj_rec(float) : the adjacency recall
    '''

    var_names = true_G_pd.columns
    true_G = true_G_pd.to_numpy()
    est_G = est_G_pd.to_numpy()
    n_nodes = true_G.shape[0]

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    true_positive_edges = []
    false_negative_edges = []
    false_positive_edges = []
    true_negative_edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            # adjacent in true_G
            if true_G[i, j] != 0 and true_G[j, i] != 0:

                # adjacent in G2
                if est_G[i, j] != 0 and est_G[j, i] != 0:
                    tp += 1
                    true_positive_edges.append([var_names[i], var_names[j]])

                # not adjacent in G2
                else:
                    fn += 1
                    false_negative_edges.append([var_names[i], var_names[j]])

            # not adjacent in true_G
            else:
                # adjacent in est_G
                if est_G[i, j] != 0 and est_G[j, i] != 0:
                    fp += 1
                    false_positive_edges.append([var_names[i], var_names[j]])

                # not adjacent in G2
                else:
                    tn += 1
                    true_negative_edges.append([var_names[i], var_names[j]])

    if tp + fp != 0:
        adj_prec = tp / (tp + fp)
    else:
        adj_prec = np.NaN

    if tp + fn != 0:
        adj_rec = tp / (tp + fn)
    else:
        adj_rec = np.NaN

    return adj_prec, adj_rec