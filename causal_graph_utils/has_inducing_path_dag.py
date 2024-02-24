
import numpy as np
from AutoCD.causal_graph_utils.is_collider import *

def has_inducing_path_dag(X, Y, dag, is_ancestor, is_latent, verbose=False):

    '''
    Checks if nodes X and Y are connected in the dag
    with an inducing path wrt a set of latent variables L.

    A path is inducing relative to a set of nodes L if (Borbudakis et al 2012):
        - every non-endpoint vertex on p is either in L or a collider
        AND
        - every collider on p is an ancestor of an end-point vertex of the path

    Author: kbiza@csd.uoc.gr, based on matlab code by striant@csd.uoc.gr

    Args:
        X (int): the node X
        Y (int): the node Y
        dag (numpy array): the matrix of the DAG
                           dag(i, j) = 2 and dag(j, i) = 3: i-->j
        is_ancestor (numpy array):  boolean array
                                    is_ancestor(i,j)=True if i is ancestor of j in a dag
        is_latent(numpy vector): boolean
                                is_latent[i]=True if i is latent variable
        verbose (bool): print if True

    Returns:
        has_ind_path (bool) : True if X and Y are connected in the DAG with an inducing path
    '''

    n_nodes = dag.shape[1]
    visited = np.zeros((n_nodes, n_nodes), dtype=bool)
    Q = np.zeros((n_nodes * n_nodes, 2), dtype=int)

    visited[:, X] = True
    visited[Y, :] = True

    # Initialize Q by adding neighbors of X
    neighbors = np.where(np.logical_and(dag[X, :] != 0, dag[:, X] != 0))[0]
    neighbors = neighbors.tolist()
    n_neighbors = len(neighbors)

    if n_neighbors != 0:
        visited[X, neighbors] = True
        Q[0:n_neighbors, 0] = X
        Q[0:n_neighbors, 1] = neighbors
        curQ = n_neighbors
    else:
        curQ = 0


    while (curQ):

        curX = Q[curQ - 1, 0]
        curY = Q[curQ - 1, 1]
        curQ = curQ - 1

        neighbors = []
        for i in range(n_nodes):

            if curX == i:
                continue

            # if visited
            if visited[curY, i]:
                continue

            # if no edge
            if np.logical_and(dag[curY, i] == 0, dag[i, curY] == 0):
                continue

            if verbose:
                print('Testing triple %d-%d-%d\n' % (curX, curY, i))

            if np.logical_or(np.logical_and(is_latent[curY], not is_collider(curX, curY, i, dag)),
                             (np.logical_and(is_collider(curX, curY, i, dag), is_ancestor[curY, [X, Y]].any()))):

                if verbose:
                    print('\t latent or possible colliders, adding %d to neighbors\n' % (i))

                neighbors = neighbors + [i]

                if i == Y:
                    has_ind_path = True
                    return has_ind_path

                continue

        n_neighbors = len(neighbors)
        if n_neighbors != 0:
            visited[curY, neighbors] = True
            Q[curQ: curQ + n_neighbors, 0] = curY
            Q[curQ: curQ + n_neighbors, 1] = neighbors
            curQ = curQ + n_neighbors

    has_ind_path = False
    return has_ind_path
