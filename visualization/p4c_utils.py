
import py4cytoscape as p4c
import re

# Utility functions for Cytoscape
# Author: kbiza@csd.uoc.gr

def p4c_find_edges_in_paths(path_set, net_suid):

    '''
    Select edges of specific paths in cytoscape
    Based on 'select_edges_connecting_selected_nodes' from py4cytoscape
    Parameters
    ----------
        path_set(list): list of paths
        net_suid(int): the network id

    Returns
    -------
        res(list): selected edges
    '''

    all_edges = p4c.get_all_edges(net_suid)

    selected_sources = set()
    selected_targets = set()

    for path in path_set:
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]

            selected_sources |= set(filter(re.compile('^' + node_i + ' ' +'.*'+ node_j + '$').search, all_edges))
            selected_targets |= set(filter(re.compile('^' + node_j + ' ' +'.*'+ node_i + '$').search, all_edges))

    selected_edges = list(selected_sources.union(selected_targets))

    if len(selected_edges) == 0: return None
    res = p4c.select_edges(selected_edges, by_col='name', preserve_current_selection=False, network=net_suid)

    return res


def p4c_select_in_path(path_set, net_suid):

    '''
    Select nodes and edges in cytoscape
    Parameters
    ----------
        path_set(list): list of paths
        net_suid(int): the id of the network

    Returns
    -------
        returns selected nodes and edges
    '''

    nodes_ = []
    p4c.clear_selection()
    for path in path_set:
        sel_nodes = p4c.select_nodes(path, network=net_suid, by_col='name')
        nodes_ = nodes_ + sel_nodes['nodes']

    sel_edges = p4c_find_edges_in_paths(path_set, net_suid)

    p4c.clear_selection()
    if sel_edges:
        return sel_edges['nodes'], sel_edges['edges']
    else:
        return [], []


def p4c_subnetwork(nodes_id, edges_id, net_suid, net_name):

    '''
    Creates a new graph in cytoscape
    Parameters
    ----------
        nodes_id(list): nodes ids in cytoscape
        edges_id(list): edges ids in cytoscape
        net_suid(int): network id in cytoscape
        net_name(str): network name

    Returns
    -------
        subnet_id(int): the new network id
    '''

    if not nodes_id:
        return None

    nodes_sel = p4c.select_nodes(nodes_id, network=net_suid)
    edges_sel = p4c.select_edges(edges_id, network=net_suid)
    subnet_id = p4c.create_subnetwork(nodes=nodes_sel['nodes'],  nodes_by_col='name',
                                                  subnetwork_name=net_name,
                                                  network=net_suid)

    p4c.clear_selection(network=net_suid)

    return subnet_id