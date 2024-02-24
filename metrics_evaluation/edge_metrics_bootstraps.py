import numpy as np
import pandas as pd


# Functions for computing edge consistency and edge similarity using bootstrapped graphs
# Author: kbiza@csd.uoc.gr

def is_consistent_edge(m1_ij, m1_ji, m2_ij, m2_ji):
    '''
    Checks if two edges are consistent
    Author: kbiza@csd.uoc.gr
    Args:
        m1_ij(int):  notation of matrix1[i,j]
        m1_ji(int):  notation of matrix1[j,i]
        m2_ij(int):  notation of matrix2[i,j]
        m2_ji(int):  notation of matrix2[j,i]

    Returns:
        is consistent(bool) : True or False
    '''

    # identical edges (or identical absence of edge)
    if m1_ij == m2_ij and m1_ji == m2_ji:
        is_consistent = True

    # consistent edges
    else:
        # i o-o j  is consistent with  io->j, i<->j, i-->j,  i<--j, i<-oj
        if m1_ij == 1 and m1_ji == 1 and m2_ij != 0 and m2_ji != 0:
            is_consistent = True

        # i o-> j  is consistent with  i<->j, i-->j, i o-o j
        elif m1_ij == 2 and m1_ji == 1:
            if m2_ij == 2 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 2 and m2_ji == 3:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-o j  is consistent with  i<->j, i<--j, i o-o j
        elif m1_ij == 1 and m1_ji == 2:
            if m2_ij == 2 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 3 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent=False

        # i --> j is consistent with  io->j, i o-o j
        elif m1_ij == 2 and m1_ji == 3:
            if m2_ij == 2 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-- j  is consistent with  i<-oj, i o-o j
        elif m1_ij == 3 and m1_ji == 2:
            if m2_ij == 1 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-> j  is consistent with  io-oj  io->j, i<-oj
        elif m1_ij == 2 and m1_ji == 2:
            if m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 2 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 2:
                is_consistent = True
            else:
                is_consistent = False

        # no edge in m1, edge in m2
        elif m1_ij==0 and m2_ij!=0:
            is_consistent = False

        # edge in m1, no edge in m2
        elif m1_ij != 0 and m2_ij == 0 :
            is_consistent = False

        else:
            print("problem with notation")
            is_consistent = False

    return is_consistent


def edge_metrics_on_bootstraps(best_mec_matrix, bootstrapped_mec_matrix):
    '''
    Args:
        best_mec_matrix(pandas Dataframe) : matrix of a mec graph
        bootstrapped_mec_matrix(list of pandas Dataframes): the bootstrapped mec graphs

    Returns:
        edge_data_pd(pandas dataframe): of size (number of edges)*(3) which contains information about all edges
            # 1st column : source
            # 2nd column : target
            # 3rd column : consistency count
            # 4th column : discovery count
            edge_consistency : percentage of consistent edges in the bootstrapped graphs
            edge_discovery : percentage of identical edges in the bootstrapped graphs

        matrix_consistency_pd(pandas dataframe):
                percentage of consistent edges in the bootstrapped graphs in a matrix form
    '''

    n_bootstraps = len(bootstrapped_mec_matrix)

    row_names = best_mec_matrix.columns.to_list()
    column_names = best_mec_matrix.columns.to_list()

    n_nodes = best_mec_matrix.shape[0]
    n_edges = int(np.count_nonzero(best_mec_matrix) / 2)

    edge_data = np.empty((n_edges, 4), dtype='object')
    matrix_consistency_data = np.zeros((n_nodes, n_nodes), dtype='float')
    c = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if best_mec_matrix.iloc[i, j] != 0:

                edge_consistency = 0
                edge_discovery = 0
                for nb in range(n_bootstraps):

                    # consistent edges
                    if is_consistent_edge(best_mec_matrix.iloc[i, j], best_mec_matrix.iloc[j, i],
                                            bootstrapped_mec_matrix[nb].iloc[i, j], bootstrapped_mec_matrix[nb].iloc[j, i]):
                        edge_consistency += 1

                    # same edges
                    if bootstrapped_mec_matrix[nb].iloc[i, j] == best_mec_matrix.iloc[i, j] and \
                            bootstrapped_mec_matrix[nb].iloc[j, i] == best_mec_matrix.iloc[j, i]:
                        edge_discovery += 1

                edge_data[c] = [row_names[i], column_names[j], edge_consistency/n_bootstraps, edge_discovery/n_bootstraps]
                c += 1
                matrix_consistency_data[i, j] = edge_consistency/n_bootstraps
                matrix_consistency_data[j, i] = edge_consistency / n_bootstraps

    edge_data_pd = pd.DataFrame(data=edge_data, columns=['source', 'target', 'edge_consistency', 'edge_discovery'])
    matrix_consistency_pd = pd.DataFrame(matrix_consistency_data,
                                         columns=best_mec_matrix.columns,
                                         index=best_mec_matrix.columns )

    return edge_data_pd, matrix_consistency_pd
