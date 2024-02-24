import numpy as np

def bidirected_path(i, matrix):
    '''
    Recursive function to find the nodes that are reachable in any bidirected path starting from the node i
    Author : kbiza@csd.uoc.gr
    Args:
        i (int): the starting node (not a list of integer!!)
        matrix (numpy array): matrix of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j

    Returns:
        list_nodes (list): the nodes that are reachable in any bidirected path starting from node i
    '''

    bidirected_neighbors = np.where(np.logical_and(matrix[i, :] == 2, matrix[:, i] == 2))[0]
    bidirected_neighbors = bidirected_neighbors.tolist()

    if len(bidirected_neighbors) == 0:
        return

    list_nodes = []
    list_nodes = list_nodes+bidirected_neighbors

    matrix[i, :] = 0
    matrix[:, i] = 0

    for j in bidirected_neighbors:
        next_neighbors = bidirected_path(j, matrix)
        if next_neighbors:
            list_nodes = list_nodes+next_neighbors

    return list_nodes
