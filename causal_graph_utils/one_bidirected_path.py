import numpy as np

def one_bidirected_path_from_to(matrix, start, end, path_=[]):

    '''
    Recursive function to search for at least one bidirected path  between 'start' node and 'end' node
    Author : kbiza@csd.uoc.gr
        Args:
            matrix(numpy array) : matrix of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j
            start (int):  the first node in the path
            end (int):  the last node in the path
            path_ (list): only needed for the recursive call (the path under search)

        Returns:
            path(list) : a list of nodes we visit from start node to end node in a bidirected path
        '''

    path_ = path_ + [start]
    if start == end:
        return path_

    neighbors = np.where(np.logical_and(matrix[start, :] == 2, matrix[:, start] == 2))[0]
    neighbors = neighbors.tolist()

    if len(neighbors) == 0:
        return []

    if end in neighbors:
        path = one_bidirected_path_from_to(matrix, end, end, path_)
        return path
    else:
        for node in neighbors:
            if node not in path_:
                path = one_bidirected_path_from_to(matrix, node, end, path_)
                if path:
                    return path
        return None
