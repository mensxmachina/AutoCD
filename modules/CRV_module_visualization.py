
from AutoCD.visualization.plot_with_cytoscape import *


def CRV_module_visualization( graph_pd, net_name, target_name, edge_info=None, edge_weights=None, n_lags=None):

    '''
    Visualizes the graph using Cytoscape.
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        graph_pd(pandas Dataframe): : the matrix of the graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs
        net_name(str) : a name for the graph
        target_name(str): the name of the target in the graph
        edge_info(pandas Dataframe or None): the estimated edge consistency and edge discovery frequencies
        edge_weights(str) : 'consistency' or 'discovery'
        n_lags(int or None): the maximum number of previous time lags in case of a time-lagged graph

    Returns
        suid(int): the network id for Cytoscape
    '''

    collection_name = 'example'
    suid, nodes_in_lags = plot_with_cytoscape(graph_pd, net_name, collection_name, target_name=target_name,
                                              n_lags=n_lags, edge_info=edge_info, edge_weights=edge_weights)

    if isinstance(n_lags, int):
        x_locations = np.arange(0, (n_lags+1)*150, 150, dtype=int)
        x_locs = x_locations.tolist()
        x_locs.reverse()

        # set nodes coordinates in time lags
        for lag in range(n_lags + 1):
            cur_nodes_ = nodes_in_lags[lag]
            cur_nodes_idx = [graph_pd.columns.get_loc(node) for node in cur_nodes_]
            cur_nodes_idx.sort()
            cur_nodes = graph_pd.columns[cur_nodes_idx].tolist()
            x_loc = [x_locs[lag]] * len(cur_nodes)
            x_loc = [x+(i%2)*40 for i, x in enumerate(x_loc)]
            y_loc_np = np.arange(1, len(cur_nodes) + 1) * (70+lag)
            y_loc = y_loc_np.tolist()
            p4c.set_node_position_bypass(cur_nodes,
                                         new_x_locations=x_loc,
                                         new_y_locations=y_loc,
                                         network=suid)

        p4c.fit_content(network=suid)
        p4c.clear_selection()
        # p4c.clear_node_property_bypass(cur_nodes, 'NODE_X_LOCATION')
        # p4c.clear_node_property_bypass(cur_nodes, 'NODE_Y_LOCATION')
    return suid
