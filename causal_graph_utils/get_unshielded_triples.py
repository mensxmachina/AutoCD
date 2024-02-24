import numpy as np

def get_unshielded_triples(G):

    '''
    Find the unshielded triples of each node in a graph G
    Author : kbiza@csd.uoc.gr based on matlab code by striant@csd.uoc.gr
    Parameters
    ----------
        G(numpy array): the matrix of the graph
    Returns
    -------
        unshielded_triples(dictionary of list) : each key corresponds to a node and
                                                each value contains two lists with the matrix coordinates (x,y)
                                                of the unshielded triples
    '''

    n_nodes = G.shape[1]
    unshielded_triples = {}

    for i in range(n_nodes):
        neighbours = np.where(np.transpose(G[i,:]))[0]
        # neighbours = neighbours.tolist()
        r1 = np.triu(G[np.ix_(neighbours,neighbours)]==0)
        r2 = np.eye(len(neighbours), dtype=int) == 0
        r3 = np.logical_and(r1, r2)
        [x,y] = np.nonzero(r3)
        unshielded_triples[i] = [neighbours[x], neighbours[y]]

    return unshielded_triples

