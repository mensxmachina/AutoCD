
import pandas as pd
import numpy as np

def output_to_array(t_graph, var_names):

    """
    Converts tigramite graph to matrix of time-lagged causal graph
            i.e. every variable appears in all time lags

        Parameters
        ----------
        t_graph (Tigramite object) : output graph from tigramite (time series graph)

        Returns
        -------
        matrix_pd(pandas Dataframe): matrix of size N*N where N is the number of nodes over all time lags
            matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
            matrix(i, j) = 1 and matrix(j, i) = 1: io-oj   in PAGs
            matrix(i, j) = 2 and matrix(j, i) = 2: i<->j   in MAGs and PAGs
            matrix(i, j) = 3 and matrix(j, i) = 3: i---j   in PDAGs
            matrix(i, j) = 2 and matrix(j, i) = 1: io->j
    """

    # t_graph = output['graph']
    n_nodes = t_graph.shape[0]
    T = t_graph.shape[2]

    matrix = np.zeros((n_nodes * T, n_nodes * T), dtype=int)

    for step in range(T):
        for i in range(n_nodes):
            for j in range(n_nodes):

                if t_graph[i, j, step] != '':

                    for t in range(step, T):
                        i_ = n_nodes * t + i
                        j_ = n_nodes * (t - step) + j

                        edge = t_graph[i, j, step]

                        if edge == 'o-o':
                            matrix[i_, j_] = 1
                            matrix[j_, i_] = 1
                        elif edge == '-->':
                            matrix[i_, j_] = 2
                            matrix[j_, i_] = 3
                        elif edge == '<--':
                            matrix[j_, i_] = 2
                            matrix[i_, j_] = 3
                        elif edge == '<->':
                            matrix[i_, j_] = 2
                            matrix[j_, i_] = 2
                        elif edge == 'o->':
                            matrix[i_, j_] = 2
                            matrix[j_, i_] = 1
                        elif edge == '<-o':
                            matrix[j_, i_] = 2
                            matrix[i_, j_] = 1
                        elif edge == 'x-x':
                            matrix[i_, j_] = 1
                            matrix[j_, i_] = 1
                        elif edge == 'x->':
                            matrix[i_, j_] = 2
                            matrix[j_, i_] = 1
                        elif edge == '<-x':
                            matrix[j_, i_] = 2
                            matrix[i_, j_] = 1

                        else:
                            raise ValueError('%s edge not included' % edge)

    matrix_pd = pd.DataFrame(matrix, columns=var_names, index=var_names)

    return matrix_pd