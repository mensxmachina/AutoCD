
import py4cytoscape as p4c
from py4cytoscape import gen_edge_color_map
from py4cytoscape import palette_color_brewer_s_Greys
import itertools
from AutoCD.visualization.matrix_to_cyto import *
from AutoCD.visualization.p4c_utils import *


def visualize_graph(matrix_pd, net_name, collection_name, graph_type,
                   source_net=None,  target_name=None,
                   edge_info=None, edge_weights=None,  show_weights=True,
                   exposure=None, adj_set=None):


    '''
    Visualizes the graph using Cytoscape.
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        matrix_pd(pandas Dataframe): the matrix of the graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs or i---j in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
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
    '''


    colors = {'all_nodes': '#E4E3E3',
              'light_green': '#DEEAB4',
              'green': '#AAD6C2',
              'yellow': '#F8E86E',
              'pink': '#BA6B87',
              'blue': '#7097AD',
              'green2': '#BACEA3',
              'tomato': '#E77975',
              'orange': '#FDC619',
              'light_orange': '#FDD49E',
              'grey_pink': '#F4E7F3',
              'grey_lila':'#ECE7F2'}


    p4c.cytoscape_ping()
    p4c.cytoscape_version_info()

    cyto_edges = matrix_to_cyto(matrix_pd,graph_type)
    if isinstance(edge_info, pd.DataFrame):
        cyto_edges.insert(cyto_edges.shape[1], "edge_consistency", edge_info['edge_consistency'])
        cyto_edges.insert(cyto_edges.shape[1], "edge_discovery", edge_info['edge_discovery'])

    net_suid = p4c.create_network_from_data_frames(edges=cyto_edges, title=net_name, collection=collection_name)

    # Style settings
    node_size = 35
    sel_node_size=45
    node_shape = 'ellipse'
    node_color = colors.get('all_nodes')
    edge_transparency = 150

    style_name = str(np.random.randint(0,100, size=1)[0])

    # Apply style settings
    defaults = {'NODE_SHAPE': node_shape, 'NODE_SIZE': node_size, 'EDGE_TRANSPARENCY': edge_transparency}

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


    # Customize selected nodes
    if isinstance(target_name, list):
        for t in target_name:
            if np.count_nonzero(matrix_pd.loc[:, t] > 0):
                p4c.set_node_color_bypass(t, colors.get('green'))
                p4c.set_node_border_color_bypass(t, colors.get('green'))
                p4c.set_node_size_bypass(t, sel_node_size)
                p4c.set_node_shape_bypass(t, ['RECTANGLE'])
    elif isinstance(target_name, str):
            if np.count_nonzero(matrix_pd.loc[:, target_name] > 0):
                p4c.set_node_color_bypass(target_name, colors.get('green'))
                p4c.set_node_border_color_bypass(target_name, colors.get('green'))
                p4c.set_node_size_bypass(target_name, sel_node_size)
                p4c.set_node_shape_bypass(target_name, ['RECTANGLE'])

    if isinstance(exposure, list):
        # if exposure is not connected with any other node cytoscape will not show it
        exposure_=exposure.copy()
        for e in exposure:
            if np.count_nonzero(matrix_pd.loc[:, e]) == 0 :
                exposure_.remove(e)

        shape_exposure = ['OCTAGON'] * len(exposure_)
        p4c.set_node_color_bypass(exposure_, colors.get('tomato'))
        p4c.set_node_border_color_bypass(exposure_, colors.get('tomato'))
        p4c.set_node_shape_bypass(exposure_, shape_exposure)
        p4c.set_node_size_bypass(exposure_, sel_node_size)

    if isinstance(adj_set, list):
        # if a node is not connected with any other node cytoscape will not show it
        adj_set_=adj_set.copy()
        for a in adj_set:
            if np.count_nonzero(matrix_pd.loc[:, a]) == 0 :
                adj_set_.remove(a)
        shape_adjset = ['DIAMOND'] * len(adj_set)
        p4c.set_node_shape_bypass(adj_set_, shape_adjset)
        p4c.set_node_color_bypass(adj_set_, colors.get('yellow'))
        p4c.set_node_border_color_bypass(adj_set_, colors.get('yellow'))
        p4c.set_node_size_bypass(adj_set_, sel_node_size)


    if edge_weights:
        # weight_range = np.linspace(start=0, stop=1, num=10)
        size_range = np.linspace(1, 10, num=10)
        # width_range = np.linspace(start=0.1, stop=2, num=10)
        width_range = np.array([0.01, 0.1, 0.5, 0.8, 1,   1.2, 1.8,   3,  4])
        weight_range = np.array([0,   0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])


        if show_weights:
            p4c.set_edge_label_mapping(edge_weights, style_name=style_name)
            p4c.set_edge_font_size_mapping(edge_weights,
                                           weight_range.tolist(),
                                           size_range.tolist(), 'c', style_name=style_name)

        p4c.set_edge_line_width_mapping(edge_weights,
                                        weight_range.tolist(),
                                        width_range.tolist(), 'c', style_name=style_name)

        p4c.set_edge_color_mapping(**gen_edge_color_map(edge_weights,
                                                              palette_color_brewer_s_Greys(),
                                                              mapping_type='c',
                                                              style_name=style_name))

    if source_net:
        p4c.layout_copycat(source_net, net_suid,  source_column='name', target_column='name', select_unmapped=False,
                           grid_unmapped=False)
        p4c.fit_content(network=source_net)
    else:
        p4c.layout_network('force-directed', net_suid)
        p4c.fit_content(network=net_suid)

    p4c.clear_selection()

    return net_suid


def visualize_subpgraphs(paths, net_suid, path_confidence=None, separate_paths=True):

    '''
    Create subgraphs with group of paths
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        paths(dictionary): the extracted paths
        net_suid(int): the network id
        path_confidence(dictionary): path consistency or path discovery or None
        separate_paths(bool): if True then create one subgraph for each path,
                                otherwise create subgraph with group of paths

    Returns
    -------
        output in Cytoscape
    '''

    path_colors = {'blocking': '#1C6D91',
                   'confounding': '#848D2F',
                   'noncausal': '#0199A0',
                   'directed': '#99000D',
                   'potentially': '#CC4C02'}

    nodes={}
    edges={}
    for key_path in paths.keys():
        p4c.clear_selection()
        if not paths[key_path]:
            continue

        if separate_paths:
            nodes[key_path]=[]
            edges[key_path]=[]
            for i, path in enumerate(paths[key_path]):
                nodes_, edges_ = p4c_select_in_path([path], net_suid)
                nodes[key_path].append(nodes_)
                edges[key_path].append(edges_)
                p4c.clear_selection()

        else:
            nodes[key_path], edges[key_path] = p4c_select_in_path(paths[key_path], net_suid)


    # create the subgraphs for each path group
    sub_net_ids = {}
    for key_path in paths.keys():
        if not paths[key_path] or key_path == 'all':
            continue

        if separate_paths:
            sub_net_ids[key_path] = []
            for i, path in enumerate(paths[key_path]):
                sub_net_id = p4c_subnetwork(nodes[key_path][i], edges[key_path][i], net_suid, key_path)
                p4c.set_edge_color_bypass(edges[key_path][i], path_colors[key_path], network=sub_net_id)
                sub_net_ids[key_path].append(sub_net_id)
        else:
            sub_net_id = p4c_subnetwork(nodes[key_path], edges[key_path], net_suid, key_path)
            p4c.set_edge_color_bypass(edges[key_path], path_colors[key_path], network=sub_net_id)
            sub_net_ids[key_path] = sub_net_id


    # add path consistency as weight above one edge in each path
    if path_confidence:

        for key_path in paths.keys():
            if not paths[key_path] or key_path == 'all':
                continue
            p4c.clear_edge_bends()
            p4c.clear_selection()


            # add label (e.g., path consistency)
            visited = []
            for i, path in enumerate(paths[key_path]):
                j = int(np.floor(len(path)/2)-1) #0
                pair = path[j: j + 2]
                while pair in visited:
                    j += 1
                    pair = path[j: j + 2]
                visited.append(pair)

                sel_edges = p4c_find_edges_in_paths([pair], net_suid)
                edge = sel_edges['edges'][0]

                # if edge not in edges[key_path]:
                #     print('error')

                if separate_paths:
                    p4c.set_edge_label_bypass(edge, str(path_confidence[key_path][i]), network=sub_net_ids[key_path][i])
                    p4c.set_edge_font_size_bypass(edge, 20, network=sub_net_ids[key_path][i])
                else:
                    p4c.set_edge_label_bypass(edge, str(path_confidence[key_path][i]), network=sub_net_ids[key_path])
                    p4c.set_edge_font_size_bypass(edge, 20, network=sub_net_ids[key_path])
                p4c.clear_edge_bends()


    # Subgraph with all group of paths (without labels)
    if separate_paths:
        sub_all_net_id = p4c_subnetwork(list(itertools.chain.from_iterable(nodes['all'])), list(itertools.chain.from_iterable(edges['all'])), net_suid, 'all')
    else:
        sub_all_net_id = p4c_subnetwork(nodes['all'],edges['all'], net_suid, 'all')
    for key_path in paths.keys():
        if not paths[key_path] or key_path == 'all':
            continue
        if separate_paths:
            p4c.set_edge_color_bypass(list(itertools.chain.from_iterable(edges[key_path])), path_colors[key_path], network=sub_all_net_id)
        else:
            p4c.set_edge_color_bypass(edges[key_path], path_colors[key_path], network=sub_all_net_id)

    return