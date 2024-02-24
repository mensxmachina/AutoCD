import py4cytoscape as p4c
from AutoCD.visualization.matrix_to_cyto import *


def plot_with_cytoscape(matrix_pd, net_name, collection_name,
                   target_name=None, n_lags=None, source_net=None, edge_info=None, edge_weights=None):

    '''
    Visualizes the graph using Cytoscape.
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        matrix_pd(pandas Dataframe): the matrix of the graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs
        net_name(str) : a name for the graph
        collection_name : a name for the collection (needed for Cytoscape)
        target_node (str or None) : the name of the target in the graph
        source_net (int or None) : a Cytoscape network ID that has been already plotted
                if the new network has the same variables, Cytoscape will use the previous coordinates for the nodes
        edge_info (pandas Dataframe) : the values of edge consistency frequency and edge discovery frequency
        show_edge_weights (str or None): 'consistency' or 'discovery' to show as weights above the edges


    Returns
    -------
        net_suid (int) : the network id
        nodes_in_lags (list or None) : node names in each time lag
    '''

    colors = {
              'all_nodes': '#E4E3E3', #FAF9E1',
              'light_green': '#DEEAB4',
              'green': '#AAD6C2',
              'yellow': '#F8E86E'}


    p4c.cytoscape_ping()
    p4c.cytoscape_version_info()

    cyto_edges = matrix_to_cyto(matrix_pd)
    if isinstance(edge_info, pd.DataFrame):
        cyto_edges.insert(cyto_edges.shape[1], "edge_consistency", edge_info['edge_consistency'])
        cyto_edges.insert(cyto_edges.shape[1], "edge_discovery", edge_info['edge_discovery'])

    net_suid = p4c.create_network_from_data_frames(edges=cyto_edges, title=net_name, collection=collection_name)

    # Style settings
    node_size = 35
    node_shape = 'ellipse'
    node_color = colors.get('all_nodes')
    edge_transparency = 150

    style_name = str(np.random.randint(0,100, size=1)[0])

    # Apply style settings
    defaults = {'NODE_SHAPE': node_shape, 'NODE_SIZE': node_size, 'EDGE_TRANSPARENCY': edge_transparency}  # , #'EDGE_TRANSPARENCY': edge_transparency

    p4c.create_visual_style(style_name, defaults=defaults)

    p4c.set_node_color_default(node_color, style_name=style_name)
    p4c.set_node_border_color_default(node_color, style_name=style_name)

    p4c.set_node_label_mapping('name', style_name=style_name)

    # Edges
    p4c.set_edge_target_arrow_shape_mapping(
        'interaction_type',
        table_column_values=['Circle-Arrow', 'Arrow-Circle',
                             'Circle-Tail', 'Tail-Circle',
                             'Arrow-Tail', 'Tail-Arrow',
                             'Arrow-Arrow', 'Circle-Circle', 'Tail-Tail'],
        shapes=['ARROW', 'CIRCLE', 'NONE', 'CIRCLE', 'NONE', 'ARROW', 'ARROW', 'CIRCLE', 'NONE'],
        style_name=style_name)

    p4c.set_edge_source_arrow_shape_mapping(
        'interaction_type',
        table_column_values=['Circle-Arrow', 'Arrow-Circle',
                             'Circle-Tail', 'Tail-Circle',
                             'Arrow-Tail', 'Tail-Arrow',
                             'Arrow-Arrow', 'Circle-Circle', 'Tail-Tail'],
        shapes=['CIRCLE', 'ARROW', 'CIRCLE', 'NONE', 'ARROW', 'NONE', 'ARROW', 'CIRCLE', 'NONE'],
        style_name=style_name)


    p4c.set_visual_style(style_name)

    # apply a layout to the whole network
    p4c.layout_network('attributes-layout', network=net_suid)

    if source_net:
        p4c.layout_copycat(source_net, net_name)


    if target_name:
        if np.count_nonzero(matrix_pd.loc[:, target_name] > 0):
            p4c.set_node_color_bypass(target_name, colors.get('green'))

    if edge_weights == 'consistency':

        # p4c.set_edge_label_mapping('edge_consistency', style_name=style_name)
        edge_width_mapping = {'input_values': [0, 0.01, 0.5, 0.7, 1.0],
                              'width_values': [0.4, 0.8, 1, 2, 3]}
        p4c.set_edge_line_width_mapping('edge_consistency', edge_width_mapping['input_values'],
                                        edge_width_mapping['width_values'], 'c', style_name=style_name)

        p4c.set_edge_font_size_bypass(None, 20)

    if edge_weights == 'discovery':
        p4c.set_edge_label_mapping('edge_discovery', style_name=style_name)


    if isinstance(n_lags, int):
        nodes_in_lags = []
        for lag in range(n_lags+1):
            if lag==0:
                cur_lag = p4c.create_column_filter('filtert0', 'name', ':',
                                                   'DOES_NOT_CONTAIN',
                                                    caseSensitive=False, anyMatch=True, apply=True)
            else:
                cur_lag = p4c.create_column_filter('filtertL', 'name', ':'+str(lag),
                                                    'CONTAINS',
                                                    caseSensitive=False, anyMatch=True, apply=True)

            nodes_in_lags.append(cur_lag['nodes'])
    else:
        nodes_in_lags=None


    return net_suid, nodes_in_lags

