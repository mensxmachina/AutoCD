import numpy as np
from AutoCD.causal_graph_utils.bidirected_path import bidirected_path


def markov_boundary(target, matrix_pd):

    '''
    Identify the markov boundary of the target node.
    Function for DAGs and MAGs
    Author:kbiza@csd.uoc.gr
    Args:
        target (int): index of the target node in the matrix (not a list of int!!)
        matrix_pd (pandas Dataframe): an array of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j    in DAGs and MAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs

    Returns:
        markov_boundary (list) : list of indexes for the markov boundary ot the target

    '''

    matrix = matrix_pd.to_numpy()
    # check if the input matrix is PAG
    if np.where(matrix == 1)[0].size > 0:
        raise ValueError('cannot find MB due to an undirected edge (need DAG or MAG)')

    # Common for DAGs and MAGs
    parents = np.where(np.logical_and(matrix[target, :] == 3, matrix[:, target] == 2))[0].tolist()
    children = np.where(np.logical_and(matrix[target, :] == 2, matrix[:, target] == 3))[0].tolist()
    parents_of_children = []
    for child in children:
        parents_of_children += np.where(np.logical_and(matrix[child, :] == 3, matrix[:, child] == 2))[0].tolist()

    # District sets in MAGs
    district_i = bidirected_path(target, np.copy(matrix))
    district_children = []
    for child in children:
        district_child = bidirected_path(child, np.copy(matrix))
        if district_child:
            district_children += district_child

    parents_of_district_i = []
    if district_i:
        for di in district_i:
            parents_of_district_i += np.where(np.logical_and(matrix[di, :] == 3, matrix[:, di] == 2))[0].tolist()
    else:
        district_i =[] # bidirected_path returns none (fix if needed)

    parents_of_district_children = []
    if district_children:
        for dchild in district_children:
            parents_of_district_children += np.where(np.logical_and(matrix[dchild, :] == 3, matrix[:, dchild] == 2))[0].tolist()


    markov_boundary = parents + children + parents_of_children + \
                      district_i + district_children + \
                      parents_of_district_i + parents_of_district_children

    markov_boundary = list(set(markov_boundary))
    if target in markov_boundary:
        markov_boundary.remove(target)

    return markov_boundary
