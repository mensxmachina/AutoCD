
import pandas as pd
from AutoCD.causal_graph_utils.has_inducing_path_dag import *
from AutoCD.causal_graph_utils.find_ancestors_nx import *


def dag_to_mag_removeL(dag_pd, is_latent):

    '''
    Converts a DAG into a MAG after marginalizing out latent variables
    Author : kbiza@csd.uoc.gr based on matlab code by striant@csd.uoc.gr

    Args:
        dag_pd(pandas Dataframe): the DAG matrix
             dag(i, j) = 2 and dag(j, i) = 3: i-->j
        is_latent (numpy vector): True if variable will be marginalized out
    Returns:
        mag_pd (pandas Dataframe) : the MAG matrix
            mag(i, j) = 2 and mag(j, i) = 3: i-->j
            mag(i, j) = 2 and mag(j, i) = 2: i<->j
            mag(i, j) = 2 and mag(j, i) = 1: io->j

        mag_removeL_pd (pandas Dataframe) : the MAG matrix where we drop the columns and rows
                    that correspond to the latent variables
    '''

    n_nodes = dag_pd.shape[0]
    dag = dag_pd.to_numpy()

    # find ancestors for each node
    is_ancestor = find_ancestors_nx(dag)

    mag = np.zeros((n_nodes, n_nodes), dtype=int)

    for X in range(n_nodes):
        if is_latent[X]:
            continue

        for Y in range(X + 1, n_nodes):
            if is_latent[Y]:
                continue

            if dag[X, Y] == 2 and dag[Y, X] == 3:
                mag[X, Y] = 2
                mag[Y, X] = 3

            elif dag[Y, X] == 2 and dag[X, Y] == 3:
                mag[Y, X] = 2
                mag[X, Y] = 3

            elif has_inducing_path_dag(X, Y, dag, is_ancestor, is_latent):
                if is_ancestor[X, Y]:
                    mag[X, Y] = 2
                    mag[Y, X] = 3
                elif is_ancestor[Y, X]:
                    mag[Y, X] = 2
                    mag[X, Y] = 3
                else:
                    # print('has inducing path', dag_pd.columns[X], dag_pd.columns[Y])
                    mag[X, Y] = 2
                    mag[Y, X] = 2

    mag_removeL = mag.copy()
    mag_removeL = np.delete(mag_removeL, is_latent, axis=0)
    mag_removeL = np.delete(mag_removeL, is_latent, axis=1)

    mag_pd = pd.DataFrame(mag, columns=dag_pd.columns, index=dag_pd.columns)
    mag_removeL_pd = pd.DataFrame(mag_removeL, columns=dag_pd.columns[~is_latent], index=dag_pd.columns[~is_latent])

    return mag_pd, mag_removeL_pd