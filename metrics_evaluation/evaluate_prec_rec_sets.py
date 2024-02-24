
import numpy as np

def evaluate_prec_rec_sets(true_set, est_set):

    '''
    Computes the precision and recall of two sets
    Parameters
    ----------
        true_set(list): set with true elements
        est_set(list): set with estimated elements
    Returns
    ----------
        prec(float): precision
        rec(float): recall
    '''

    fp = list(set(est_set) - set(true_set))
    fn = list(set(true_set) - set(est_set))
    tp = list(set(true_set).intersection(est_set))

    if (len(tp) + len(fp)) != 0:
        prec = len(tp) / (len(tp) + len(fp))
    else:
        prec = np.NaN

    if (len(tp) + len(fn)) != 0:
        rec = len(tp) / (len(tp) + len(fn))
    else:
        rec = np.NaN

    return prec, rec

