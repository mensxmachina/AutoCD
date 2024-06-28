import pandas as pd
import numpy as np
import subprocess
import os

def read_adjset(csv_name, path_):

    '''
    Read the output csv file from R packages
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
    csv_name (str): the name of the file
    path_(str): the path of the file

    Returns
    -------
    adj_set(list or None): list if adjustment set exists, None if no adjustment set exists

    '''

    adjset_pd = pd.read_csv(os.path.join(path_, csv_name))

    if 'X1' in adjset_pd:
        print('adjustment set exists')
        adj_set=[]
        for i in range(adjset_pd.shape[1]):
            cur_set = adjset_pd['X'+str(i+1)].tolist()
            adj_set.append(cur_set)
            #adj_set = adjset_pd.to_numpy().reshape(-1).tolist()
    else:
        print('no adjustment set exists')
        adj_set = None

    return adj_set


def adjset_pcalg(graph_pd, graph_type, x, y):

    '''
    Run the pcalg R package to identify the adjustment set of X and Y
    Author: kbiza@csd.uoc.gr
    Change R version in line 57 if needed
    Parameters
    ----------
    graph_pd(pandas Dataframe):
    graph_type(str): {'dag', 'cpdag', 'mag', 'pag'}
    x(list): list of variable names
    y(list): list of variable names

    Returns
    -------
        canonical_set(list): the variable names of the canonical adj. set (if exists)
        minimal_set(list):: the variable names of the minimal adj. set (if exists)
    '''

    r_path = 'C:/Program Files/R/R-4.3.2/bin/Rscript'
    path_ = os.path.dirname(__file__)
    graph_name='graph_r.csv'
    graph_pd.to_csv(graph_name)

    subprocess.call([r_path, '--vanilla', os.path.join(path_, 'run_adjset_pcalg_r.R'),
                     graph_name, graph_type, str(x), str(y)], shell=True)

    canonical_set = read_adjset('canonical_pcalg.csv', path_)
    minimal_set = read_adjset('minimal_pcalg.csv', path_)

    # r indexing --> we need to subtract 1
    canonical_set = [[value - 1 for value in sublist] for sublist in canonical_set]
    minimal_set = [[value - 1 for value in sublist] for sublist in minimal_set]

    return canonical_set, minimal_set


def adjset_dagitty(graph_pd, graph_type, x_name, y_name):

    '''
    Run the dagitty R package to identify the adjustment set of X and Y
    Author: kbiza@csd.uoc.gr
    Change R version in line 92 if needed
    Args:
        graph_pd(pandas Dataframe): the graph as adjacency matrix
        graph_type(str): the type of the graph : {'dag', 'cpdag', 'mag', 'pag'}
        x_name(list): list of variable names
        y_name(list): list of variable names

    Returns:
        canonical_set(list): the variable names of the canonical adj. set (if exists)
        minimal_set(list):: the variable names of the minimal adj. set (if exists)
    '''

    r_path = 'C:/Program Files/R/R-4.3.2/bin/Rscript'
    path_ = os.path.dirname(__file__)

    graph_name = 'graph_r.csv'
    exp_name = 'exposures.csv'
    out_name = 'outcomes.csv'
    graph_pd.to_csv(os.path.join(path_, graph_name))

    x_names_pd = pd.DataFrame(np.array(x_name), columns=['x_names_dagitty'])
    y_names_pd = pd.DataFrame(np.array(y_name), columns=['y_names_dagitty'])
    x_names_pd.to_csv(os.path.join(path_, exp_name))
    y_names_pd.to_csv(os.path.join(path_, out_name))

    subprocess.call([r_path, '--vanilla', os.path.join(path_, 'run_adjset_dagitty_r.R'),
                     graph_name, graph_type, exp_name , out_name],shell=True)

    canonical_set = read_adjset('canonical_dagitty.csv', path_)
    minimal_set = read_adjset('minimal_dagitty.csv',path_)

    # it returns variable names, not indexes
    return canonical_set, minimal_set