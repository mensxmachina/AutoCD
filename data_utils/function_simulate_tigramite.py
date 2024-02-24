
import numpy as np
import pandas as pd
import tigramite
from tigramite.toymodels import structural_causal_processes as toys
from AutoCD.data_utils.function_simulate_tetrad import *
from AutoCD.causal_discovery.tigramite_graph_to_array import *


def lin(x): return x


def create_links(dag_pd, tau_max, n_lagged_parents=1, auto_coeff_=None, coeff_=None):

    '''
    Creates random links required for simulation with Tigramite
    Author: kbiza@csd.uoc.gr
    Parameters
    ----------
        dag_pd(pandas Dataframe) : pandas dataframe of the DAG
        tau_max(int) : the maximum number of time lags
        n_lagged_parents(int) : number of parents from previous lags
        auto_coeff_(float) :  specific autocorrelation coefficient, if needed
        coeff_(float): specific coefficient, if needed

    Returns
    -------
        links(dictionary):
    '''

    coeff_intervals = [0.1, 0.5]
    auto_coeff_intervals = [0.2, 0.9]

    dag_np = dag_pd.to_numpy().copy()
    n_nodes = dag_np.shape[1]
    nodes = np.arange(n_nodes)
    links = {}

    for node in range(dag_np.shape[1]):

        # auto-correlations
        if not auto_coeff_:
            auto_coeff = np.random.uniform(low=auto_coeff_intervals[0], high=auto_coeff_intervals[1])
        else:
            auto_coeff = auto_coeff_
        cur_list = [((node, -1), auto_coeff, lin)]

        # contemporaneous based on dag
        parents_idx = np.where(dag_np[:, node] == 2)[0]
        if len(parents_idx) > 0:
            for parent in parents_idx:
                if not coeff_:
                    coeff = np.random.uniform(low=coeff_intervals[0], high=coeff_intervals[1])
                    if np.random.choice([True, False]):
                        coeff = - coeff
                else:
                    coeff = coeff_
                cur_tuple = ((parent, 0), coeff, lin)
                cur_list.append(cur_tuple)

        # lagged edges
        for tau in range(1, tau_max + 1):
            if np.random.randint(3) == 1:  # probability 1/3
                if not coeff_:
                    coeff = np.random.uniform(low=coeff_intervals[0], high=coeff_intervals[1])
                    if np.random.choice([True, False]):
                        coeff = - coeff
                else:
                    coeff = coeff_
                cur_nodes = nodes.copy()
                cur_nodes = np.delete(cur_nodes, node)
                p = (1 / cur_nodes.shape[0]) * np.ones(cur_nodes.shape, dtype=float)
                lagged_parents = np.random.choice(cur_nodes, size=n_lagged_parents, replace=False, p=p)
                for lagged_pa in lagged_parents:
                    cur_tuple = ((lagged_pa, -tau), coeff, lin)
                    cur_list.append(cur_tuple)

        links[node] = cur_list


    return links



def simulate_time_series(n_nodes, n_samples,tau_max, avg_degree, max_degree, seed=None):

    '''
    Simulate temporal data using Tigramite
    Author : kbiza@csd.uoc.gr

    Parameters
    ----------
        n_nodes(int) : number of nodes
        n_samples(int) : number of samples
        tau_max(int) : maximum number of previous time lags
        avg_degree(int) :  average node degree in each time-lag
        max_degree(int) :  maximum node degree in each time-lag
        seed

    Returns
    -------
        data_pd(pandas dataframe): dataset
        time_lagged_dag(pandas dataframe): time lagged causal graph
        tigramite_graph(tigramite object): tigramite output
        links(dictionary) : the links for the simulation
    '''

    # parameters for dag (tetrad)
    simulation_type = 'GSEM'

    # parameters for tigramite
    n_lagged_parents = 1

    # create random dag
    _, _, _, dag_pd, _, _ = \
        simulate_data_tetrad(n_nodes, n_samples, avg_degree, max_degree, simulation_type, n_lags=0, seed=seed)
    var_names = dag_pd.columns

    flag = True
    while flag:
        # create random links
        links = create_links(dag_pd, tau_max, n_lagged_parents=n_lagged_parents)
        tigramite_graph = toys.links_to_graph(links=links)

        # dynamical noise term distributions: unit variance Gaussians
        noises = [np.random.randn for j in links.keys()]

        # simulate data
        data_, nonstationarity_indicator = toys.structural_causal_process(links=links, T=n_samples, noises=noises, seed=seed)
        if not nonstationarity_indicator:
            flag = False

    # Transform dag
    var_names_lagged = []
    for lag in range(tigramite_graph.shape[2]):
        if lag == 0:
            var_names_lagged.extend(var_names)
        else:
            var_names_lagged.extend([s + ':' + str(lag) for s in var_names])

    time_lagged_dag = output_to_array(tigramite_graph, var_names_lagged)
    data_pd = pd.DataFrame(data_, columns=var_names)

    return data_pd, time_lagged_dag, tigramite_graph, links