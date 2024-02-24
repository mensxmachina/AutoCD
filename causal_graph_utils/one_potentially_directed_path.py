import numpy as np

def one_potentially_directed_path(matrix, start, end,  path_=[]):
    '''
    Recursive function to search for at least one potentially directed path from 'start' node to 'end' node
    Author : kbiza@csd.uoc.gr
    Args:
        matrix(numpy array): matrix of size N*N where N is the number of nodes in tetrad_graph
            matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
            matrix(i, j) = 1 and matrix(j, i) = 1: io-oj
            matrix(i, j) = 2 and matrix(j, i) = 2: i<->j
            matrix(i, j) = 3 and matrix(j, i) = 3: i---j
            matrix(i, j) = 2 and matrix(j, i) = 1: io->j
        start(int):  the first node in the path
        end(int):  the last node in the path
        path_ (list): the path under search through the recursive functions

    Returns:
        path(list) : a list of nodes that appear in one potentially directed path from start node to end node
               - the path has not necessarily the minimum length

        Zhang Phd, 2007, page 108 :
            for every 0<=i<=n-1 the edge between Vi and Vi+1 is not into Vi nor is out of Vi+1
            intuitively : a path that could be oriented into a directed path by changing the
                          circles on the path into appropriate tails or arrowheads
    '''

    path_ = path_ + [start]
    if start == end:
        return path_

    r1 = np.logical_and(matrix[start, :] == 2, matrix[:, start] == 1)
    r2 = np.logical_and(matrix[start, :] == 2, matrix[:, start] == 3)
    r3 = np.logical_and(matrix[start, :] == 1, matrix[:, start] == 1)

    neighbors = np.where(np.logical_or(np.logical_or(r1, r2), r3))[0]

    # neighbors = np.where(np.logical_or(np.logical_and(matrix[start, :] == 2, matrix[:, start] == 1),
    #                                     np.logical_and(matrix[:, start] == 1, matrix[start, :] == 1)))[0]

    neighbors = neighbors.tolist()

    if len(neighbors) == 0:
        return []

    if end in neighbors:
        path = one_potentially_directed_path(matrix, end, end, path_)
        return path
    else:
        for node in neighbors:
            if node not in path_:
                path = one_potentially_directed_path(matrix, node, end, path_)
                if path:
                    return path
        return None
