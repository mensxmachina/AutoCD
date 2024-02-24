from AutoCD.causal_graph_utils.one_directed_path import *
from AutoCD.causal_graph_utils.one_potentially_directed_path import *
from AutoCD.causal_graph_utils.one_path_anytype import *
from AutoCD.causal_graph_utils.find_ancestors_nx import *
import py4cytoscape as p4c


def apply_edge_styles(net_id, sub_net_id):

    cur_style = p4c.get_current_style(network=net_id)
    edge_width_mapping = {'input_values': [0, 0.01, 0.5, 0.7, 1.0],
                          'width_values': [0.4, 0.8, 1, 2, 3]}
    p4c.set_edge_line_width_mapping('edge_consistency', edge_width_mapping['input_values'],
                                    edge_width_mapping['width_values'], 'c', style_name=cur_style,
                                    network=sub_net_id)

    p4c.set_edge_font_size_bypass(None, 20, network=sub_net_id)
    # p4c.set_edge_label_mapping('edge_consistency', style_name=cur_style, network=net_id)

def CRV_module_causalpaths(graph_pd, source_name, target_name, network_id):

    '''
    Creates subnetworks in Cytoscape that show the identified causal paths between two nodes (source and target)
    Parameters
    ----------
        graph_pd(pandas Dataframe) : the matrix of the causal graph
        source_name(str): the name of the source node
        target_name(str): the name of the target node
        network_id(int): the Cytoscape network id

    Returns
    ---------
        all_paths(list): the ids of the identified causal paths
    '''

    colors = {
        'all_nodes': '#FAF2D4',
        'light_green': '#DEEAB4',
        'green': '#AAD6C2',
        'yellow': '#F8E86E'}

    source_idx = graph_pd.columns.get_loc(source_name)
    target_idx=graph_pd.columns.get_loc(target_name)
    cur_style = p4c.get_current_style(network=network_id)

    # Neighbors
    symbols_ij = ('o', '>', '-', 'z')
    symbols_ji = ('o', '<', '-', 'z')
    neighbors_names = []
    print('Neighbors of the target')
    for i in range(graph_pd.shape[1]):
        if graph_pd.iloc[i, target_idx]:
            neighbors_names.append(graph_pd.columns[i])
            print('\t%s %s--%s %s' %
                  (graph_pd.columns[i], symbols_ji[graph_pd.iloc[target_idx, i] - 1],
                   symbols_ij[graph_pd.iloc[i, target_idx] - 1], graph_pd.columns[target_idx]))

    p4c.set_node_color_bypass(neighbors_names, colors.get('yellow'), network=network_id)


    # Causal ancestors
    ancestors = find_ancestors_nx(graph_pd, node=target_idx)
    ancestors_names = graph_pd.columns[ancestors].tolist()
    p4c.set_node_color_bypass(ancestors_names, colors.get('light_green'), network=network_id)


    # Causal Paths
    all_paths = {}
    # Potentially directed path
    potentially_directed_path = one_potentially_directed_path(graph_pd.to_numpy(), source_idx, target_idx)
    if isinstance(potentially_directed_path, list):
        path_names_pdir = graph_pd.columns[potentially_directed_path].tolist()
        if path_names_pdir:
            potential_dir_net = p4c.create_subnetwork(nodes=path_names_pdir, nodes_by_col='name',
                                                  subnetwork_name='potentially directed path',
                                                  network=network_id)
            p4c.fit_content(network=potential_dir_net)
            apply_edge_styles(network_id, potential_dir_net)
            all_paths['potentially_directed']=potential_dir_net

    # Directed path
    directed_path = one_directed_path(graph_pd.to_numpy(), source_idx, target_idx)
    if isinstance(directed_path, list):
        path_names_dir = graph_pd.columns[directed_path].tolist()
        if path_names_dir:
            dir_net = p4c.create_subnetwork(nodes=path_names_dir, nodes_by_col='name',
                                                  subnetwork_name='directed path',
                                                  network=network_id)
            p4c.fit_content(network=dir_net)
            apply_edge_styles(network_id, dir_net)
            all_paths['directed']=dir_net

    # Path of any type
    any_path = one_path_anytype(graph_pd.to_numpy(), source_idx, target_idx)
    if isinstance(any_path, list):
        path_names = graph_pd.columns[any_path].tolist()
        if path_names:
            net_any = p4c.create_subnetwork(nodes=path_names, nodes_by_col='name',
                                                  subnetwork_name='path of any type',
                                                  network=network_id)
            p4c.fit_content(network=net_any)
            apply_edge_styles(network_id, net_any)
            all_paths['any_type'] = net_any


    return all_paths