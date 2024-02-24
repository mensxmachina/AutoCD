
import numpy as np
import pandas as pd

from AutoCD.causal_discovery.tetrad_graph_to_array import tetrad_graph_to_array


def simulate_data_tetrad (n_nodes, n_samples, avg_degree, max_degree, simulation_type,
                          function_gsem = 'TSUM(NEW(B)*$)',
                          error_gsem = 'Normal(0, (Uniform(1, 3)))',
                          betas_gsem = 'Split(-1.0,-.5,.5,1.0)',
                          minCategories = 2, maxCategories = 4, percentDiscrete = 50, n_lags=2,  seed=None):

    '''
    Simulations using tetrad version 7.6.0
    Author : kbiza@csd.uoc.gr
    Args:
        n_nodes(int): the number of nodes
        n_samples(int):  the number of samples
        avg_degree(int): the average number of neighbors
        max_degree(int): the maximum number of neighbors
        simulation_type(str):  { GSEM, BayesNet, LeeHastie, CondGauss
        function_gsem: check more available functions in tetrad gui : Graph --> Parametric model --> GSEM
                        'TSUM(NEW(B)*$)',
                        'TSUM(NEW(B)*$^2)',
                        'TSUM(NEW(B)*abs($))',
                        'TSUM(NEW(B)*(1/$))',
                        'TSUM(NEW(B)*(ln(abs($))))',
                        'tanh(TSUM(NEW(B)*$))',
                        'TSUM(NEW(B)*signum($)*pow(abs($), 0.5))',
                        'TSUM(NEW(B)*signum($)*pow(abs($), 1.5))',
                        'TPROD(NEW(B)*$)',
                        'TSUM(NEW(B)*(ln(cosh($))))'
        error_gsem:  'Normal(0, (Uniform(1, 3)))' (check tetrad gui)
        betas_gsem:  'Split(-1.0,-.5,.5,1.0)' (check tetrad gui)
        minCategories(int): the minimum number of categories for the categorical variables
        maxCategories(int): the maximum number of categories for the categorical variables
        percentDiscrete(int): the percentage of categorical variables in a mixed dataset

    Returns:
        tDag(java/tetrad object): dag
        tCpdag(java/tetrad object): cpdag
        tData(java/tetrad object): data
        dag_pd(pandas Dataframe): matrix of the true dag
        cpdag_pd(pandas Dataframe): matrix of the true cpdag

            matrix: size N*N where N is the number of nodes in tetrad_graph
                    matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                    matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                    matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                    matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                    matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs

        data_pd(pandas Dataframe): simulated data
    '''

    from edu.cmu.tetrad import data
    from java import util
    from edu.cmu.tetrad.algcomparison import simulation
    from edu.cmu.tetrad import util
    from edu.cmu.tetrad.algcomparison import graph
    from edu.cmu.tetrad.graph import GraphTransforms

    # syn=n_edges/n_nodes
    # avg=round(2*syn)
    # max_degree=avg+1
    # avg_degree=avg

    parameters = util.Parameters()
    parameters.set("numMeasures", n_nodes)
    parameters.set("sampleSize", n_samples)
    parameters.set("avgDegree", avg_degree)
    parameters.set("maxDegree", max_degree)
    parameters.set("differentGraphs", True)
    parameters.set("numRuns", 1)
    parameters.set("numLatents", 0)
    parameters.set("randomizeColumns", False)

    if seed:
        parameters.set("seed", seed)

    G = graph.RandomForward()

    if simulation_type == 'GSEM':
        parameters.set('generalSemFunctionTemplateMeasured', function_gsem)
        parameters.set('generalSemErrorTemplate', error_gsem)
        parameters.set('generalSemParameterTemplate', betas_gsem)
        Sim = simulation.GeneralSemSimulation(G)

    elif simulation_type == 'BayesNet':
        parameters.set("minCategories", minCategories)
        parameters.set("maxCategories", maxCategories)
        Sim = simulation.BayesNetSimulation(G)

    elif simulation_type == 'LeeHastie':
        parameters.set("minCategories", minCategories)
        parameters.set("maxCategories", maxCategories)
        parameters.set("percentDiscrete", percentDiscrete)
        Sim = simulation.LeeHastieSimulation(G)

    elif simulation_type == 'CondGauss':
        parameters.set("minCategories", minCategories)
        parameters.set("maxCategories", maxCategories)
        Sim = simulation.ConditionalGaussianSimulation(G)

    # simulation for time series : higher values at later samples
    elif simulation_type == 'TimeSeries':
        parameters.set('numLags', n_lags)
        Sim = simulation.TimeSeriesSemSimulation(G)


    else :
        raise ValueError('%s simulation type not included' % simulation_type)

    Sim.createData(parameters, True)
    tDag = Sim.getTrueGraph(0)
    tData = Sim.getDataModel(0)
    # t_knowledge = Sim.getKnowledge()

    tCpdag = GraphTransforms.cpdagForDag(tDag)

    dag = tetrad_graph_to_array(tDag, n_lags=n_lags)
    cpdag = tetrad_graph_to_array(tCpdag, n_lags=n_lags)


    data_ = np.zeros((int(tData.getNumRows()), int(tData.getNumColumns())), dtype=float)
    for row in range(int(tData.getNumRows())):
        for col in range(int(tData.getNumColumns())):
            data_[row, col] = tData.getDouble(row , col)


    var_names = []

    for i in range(n_nodes):
        var_names.append('V' + str(i + 1))
    if simulation_type == 'TimeSeries':
        for lag in range(n_lags):
            for i in range(n_nodes):
                var_names.append('V' + str(i + 1) +':' + str(lag + 1))

    data_pd = pd.DataFrame(data_, columns=var_names)

    dag_pd = pd.DataFrame(dag, columns=var_names, index=var_names)
    cpdag_pd = pd.DataFrame(cpdag, columns=var_names, index=var_names)

    return tDag, tCpdag, tData, dag_pd, cpdag_pd, data_pd


