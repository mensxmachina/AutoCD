

# Functions to enforce stationarity on time-lagged causal graphs
# Author: kbiza@csd.uoc.gr


def enforce_stationarity_arrowheads(G, graph_pd, n_lags, verbose):
    '''
    # Adds arrowheads on edges that end to future time lags, e.g. Xt-1 *--> Xt
    Parameters
    ----------
        G(numpy array) : the matrix of the time-lagged graph to change
        graph_pd(pandas Dataframe) : the original matrix of the time-lagged graph
        n_lags(int) : the maximum number of previous lags
        verbose (bool)

    Returns
    -------
        G(numpy array) : the changed matrix of the time-lagged graph
    '''

    graph = graph_pd.to_numpy()
    n_nodes = int(graph_pd.shape[1] / (n_lags + 1))
    T = n_lags + 1

    for lag in range(T):
        for step in range(lag + 1, T):
            for i in range(n_nodes):
                for j in range(n_nodes):

                    i_ = n_nodes * lag + i
                    j_ = n_nodes * step + j

                    if G[j_, i_] == 1:
                        G[j_, i_] = 2
                        if verbose:
                            print('Time arrowheads: %s *-> %s ' % (graph_pd.columns[j_], graph_pd.columns[i_]))

    return G

def enforce_stationarity_tails_and_orientation(G, graph_pd, n_lags, verbose):

    '''
    Adds tails on the edges that start from the oldest time lag
        e.g. for n_lags=2,  if X2_t-1 ---> X2_t  and  X2_t-2 o--> X2_t-1
                            we set X2_t-2 ---> X2_t-1
    It also enforces stationarity inside each time lag regarding the orientation of existing edges
    Parameters
    ----------
        G(numpy array) : the matrix of the graph
        mag_pd(pandas Dataframe):
        n_lags (int) : the maximum number of previous lags
        verbose (bool)

    Returns
    -------
        G(numpy array) : the matrix of the graph
    '''

    graph = graph_pd.to_numpy()
    n_nodes = int(graph_pd.shape[1] / (n_lags + 1))
    T = n_lags + 1

    # Tails at the last time lag
    for i in range(n_nodes):
        for j in range(n_nodes):

            for lag in range(T):
                if lag + 2 < T:

                    i_cur = n_nodes * lag + i
                    j_cur = n_nodes * (lag + 1) + j

                    i_prev = n_nodes * (lag + 1) + i
                    j_prev = n_nodes * (lag + 2) + j

                    if G[i_cur, j_cur] != 0 and G[i_prev, j_prev]!=0:
                        G[i_prev, j_prev] = G[i_cur, j_cur]
                        if verbose:
                            print('Similar tails: ',
                                  graph_pd.columns[i_cur], graph_pd.columns[j_cur],
                                  graph_pd.columns[i_prev],graph_pd.columns[j_prev])

                if lag + 1 < T:

                    # check stationarity inside each time lag
                    i_cur = n_nodes * lag + i
                    j_cur = n_nodes * lag + j

                    i_prev = n_nodes * (lag + 1) + i
                    j_prev = n_nodes * (lag + 1) + j

                    if G[i_cur, j_cur] != 0 and G[i_prev, j_prev] !=0:
                        G[i_prev, j_prev] = G[i_cur, j_cur]
                        if verbose:
                            print('Similar time lags',
                                  graph_pd.columns[i_cur], graph_pd.columns[j_cur],
                                  graph_pd.columns[i_prev],graph_pd.columns[j_prev])

    return G


def enforce_stationarity_add_edge(G, mag_pd, n_lags, verbose):

    '''
    Enforces stationarity assumption on the time-lagged graph
        If At --> Bt then A_t-1 --> B_t-1  (add edge between nodes in the same time lag)
        If At-1 --> B_t then A_t-2 --> B_t-1 (add egde between nodes across time lags)
    Parameters
    ----------
        G(numpy array) : the matrix of the graph
        mag_pd(pandas Dataframe):
        n_lags (int) : the maximum number of previous lags
        verbose (bool)

    Returns
    -------
        G(numpy array) : the matrix of the graph
    '''

    mag = mag_pd.to_numpy()
    n_nodes = int(mag_pd.shape[1] / (n_lags + 1))
    T = n_lags + 1

    for i in range(n_nodes):
        for j in range(n_nodes):

            for lag in range(T):

                if lag + 1 < T:

                    # edge between nodes in the same time-lag
                    i_cur = n_nodes * lag + i
                    j_cur = n_nodes * lag + j

                    i_prev = n_nodes * (lag + 1) + i
                    j_prev = n_nodes * (lag + 1) + j

                    if G[i_cur, j_cur] != 0 and G[i_prev, j_prev] == 0:
                        G[i_prev, j_prev] = G[i_cur, j_cur]
                        G[j_prev, i_prev] = G[j_cur, i_cur]
                        if verbose:
                            print('Add edge on time lag',
                                  mag_pd.columns[i_cur], mag_pd.columns[j_cur],
                                  mag_pd.columns[i_prev],mag_pd.columns[j_prev])

    for i in range(n_nodes):
        for j in range(n_nodes):

            for lag in range(T):
                for step in range(lag + 1, T):
                    if step + 1 < T:

                        # edge between nodes in different time-lags
                        i_cur = n_nodes * lag + i           # i_cur in t
                        j_cur = n_nodes * step + j          # j_cur in t-1

                        i_prev = n_nodes * step + i         # i_prev in t-1
                        j_prev = n_nodes * (step + 1) + j   # j_prev in t-2

                        if G[i_cur, j_cur] != 0 and G[i_prev, j_prev] == 0:
                            G[i_prev, j_prev] = G[i_cur, j_cur]
                            G[j_prev, i_prev] = G[j_cur, i_cur]  # because we do not visit again this pair
                            if verbose:
                                print('Add edge across time lags',
                                      mag_pd.columns[i_cur], mag_pd.columns[j_cur],
                                      mag_pd.columns[i_prev],mag_pd.columns[j_prev])

    return G
