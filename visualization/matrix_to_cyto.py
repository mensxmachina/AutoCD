
import numpy as np
import pandas as pd


def matrix_to_cyto(matrix_pd):
    """
    Converts matrix to list of edges according to Cytoscape inputs
    Author: kbiza@csd.uoc.gr
    Args:
        matrix_pd(pandas Dataframe): matrix of size N*N where N is the number of nodes
            matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
            matrix(i, j) = 1 and matrix(j, i) = 1: io-oj  should appear only in PAGs
            matrix(i, j) = 2 and matrix(j, i) = 2: i<->j  should appear only in MAGs and PAGs
            matrix(i, j) = 3 and matrix(j, i) = 3: i---j  should appear only in PDAGs
            matrix(i, j) = 2 and matrix(j, i) = 1: io->j

    Returns:
        cyto_edges(pandas Dataframe): ontains information about all edges,  size: number of edges*3
            1st column : source
            2nd column : target
            3rd column : interaction_type
    """

    matrix=matrix_pd.to_numpy()
    row_names = matrix_pd.columns.to_list()
    column_names = matrix_pd.columns.to_list()

    n_nodes = matrix.shape[0]
    n_edges = int(np.count_nonzero(matrix) / 2)

    edge_data = np.empty((n_edges, 3), dtype='object')

    c = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if matrix[i, j] != 0 and matrix[j, i] != 0:

                if matrix[i, j] == 1:
                    iToj = 'Circle'
                elif matrix[i, j] == 2:
                    iToj = 'Arrow'
                elif matrix[i, j] == 3:
                    iToj = 'Tail'
                else:
                    raise ValueError('wrong notation on input matrix of the graph')

                if matrix[j, i] == 1:
                    jToi = 'Circle'
                elif matrix[j, i] == 2:
                    jToi = 'Arrow'
                elif matrix[j, i] == 3:
                    jToi = 'Tail'
                else:
                    raise ValueError('wrong notation on input matrix of the graph')

                interaction = jToi + '-' + iToj

                edge_data[c] = [row_names[i],  column_names[j], interaction]
                c += 1

    cyto_edges = pd.DataFrame(data=edge_data, columns=['source', 'target', 'interaction_type'])
    return cyto_edges
