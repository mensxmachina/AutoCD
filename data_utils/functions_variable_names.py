import re


def names_from_lag(varnames_lag):

    '''
    Parameters
    ----------
        varnames_lag(list): the variable names with information about time-lag
                        e.g ['V1', 'V2', 'V1:1', 'V2:1']
    Returns
    -------
        varnames(list): the variable names without lag info
                        e.g. ['V1','V2']
    '''

    all_varnames = []
    for feature, i in zip(varnames_lag, range(len(varnames_lag))):
        m1 = re.search('(^.+):', feature)
        if m1:
            all_varnames.append(m1.group(1))
        else:
            all_varnames.append(feature)
    varnames = list(set(all_varnames))


    return varnames


def lagnames_from_names(varnames, n_lags):

    '''
    Parameters
    ----------
        varnames(list): the variable names without lag info
                        e.g. ['V1','V2']
        n_lags(int): the maximum number of previous time lags
    Returns
    -------
        varnames_lag(list): the variable names with information about time-lag
                        e.g ['V1', 'V2', 'V1:1', 'V2:1']
    '''

    varnames_lagged = []
    for lag in range(n_lags + 1):
        if lag == 0:
            varnames_lagged = varnames_lagged + varnames
        else:
            cur = [s + ':' + str(lag) for s in varnames]
            varnames_lagged = varnames_lagged + cur

    return varnames_lagged