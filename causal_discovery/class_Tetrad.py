
import numpy as np
import pandas as pd
import warnings
from AutoCD.causal_graph_utils.is_dag import *


class Tetrad():

    '''
    Use the Tetrad project to run causal discovery algorithms
    Author: kbiza@csd.uoc.gr
    '''

    def __init__(self, data_pd, data_type_info, is_time_series, tiers=None):
        '''
        Update 5/6/2024: add prior knowledge and directlingam
        Parameters
        ----------
            data_pd(pandas Dataframe): the dataset
            data_type_info (pandas Dataframe) : information about the type of each variable (continuous or categorical)
            tiers(list) : sorted variables into groupings that can or cannot affect each other (based on Tetrad)
                          variables in higher-numbered tiers occur later than variables in lower-numbered tiers
                list 0 : indexes of variables without parents (exogenous)
                list 1 : a variable in this tier cannot be a cause of a variable in list 0
                list 2 : a variable in this tier cannot be a cause of a variable in list 0 or 1
            is_time_series (bool): True if dataset contains temporal variables
        '''

        self.data_pd = data_pd
        self.data_type_info = data_type_info
        self.tiers = tiers
        self.is_time_series = is_time_series

    def prepare_data(self, parameters):

        '''
        Converts the input dataset to Tetrad format
        Parameters
        ----------
            parameters(dictionary): the causal configuration parameters
        Returns
        -------
            ds(Tetrad object): the dataset
            name_map_pd (pandas Dataframe): mapping for the variable names
        '''

        from edu.cmu.tetrad import data
        from java import util


        is_cat_var = self.data_type_info['var_type'] == 'categorical'
        is_cat_var = is_cat_var.to_numpy()
        n_domain = self.data_type_info['n_domain'].to_numpy()

        data_np = self.data_pd.to_numpy()
        var_names = self.data_pd.columns.to_list()

        is_con_var = ~ is_cat_var
        my_list = util.LinkedList()
        n_samples, n_cols = np.shape(data_np)
        dataC = data_np[:, ~is_cat_var]
        dataD = data_np[:, is_cat_var]
        dataD = dataD.astype(int)

        if var_names:
            name_map = np.empty(shape=(n_cols, 3), dtype='U100')
            column_names = ['index', 'tetrad_name', 'var_name']
        else:
            name_map = np.empty(shape=(n_cols, 2), dtype='U100')
            column_names = ['index', 'tetrad_name']

        c = 0

        tetrad_names = []
        if 'n_lags' in parameters.keys():
            for lag in range(parameters['n_lags'] + 1):
                for i in range(int(n_cols / (parameters['n_lags'] + 1))):
                    if lag == 0:
                        tetrad_names.append('X' + str(i + 1))
                        # tetrad_names.append(str(i + 1))
                    else:
                        tetrad_names.append('X' + str(i + 1) + ':' + str(lag))
                        # tetrad_names.append(str(i + 1) + ':' + str(lag))
        else:
            for i in range(int(n_cols)):
                tetrad_names.append('X' + str(i + 1))


        for i in range(n_cols):

            tetrad_name = tetrad_names[i]

            if is_cat_var[i]:
                var = data.DiscreteVariable(tetrad_name, n_domain[i])
            else:
                var = data.ContinuousVariable(tetrad_name)

            my_list.add(var)
            name_map[c, 0] = c + 1
            name_map[c, 1] = tetrad_name
            if var_names:
                name_map[c, 2] = var_names[c][:]
            c += 1

        dsM = data.MixedDataBox(my_list, n_samples)

        if np.any(is_con_var):
            tdataC = np.transpose(dataC)
            dsC = data.VerticalDoubleDataBox(tdataC)
        if np.any(is_cat_var):
            tdataD = np.transpose(dataD)
            dsD = data.VerticalIntDataBox(tdataD)

        for i in range(n_samples):
            d = 0
            c = 0
            for node in range(n_cols):
                if is_con_var[node]:
                    dsM.set(i, node, dsC.get(i, c))
                    c = c + 1
                else:
                    dsM.set(i, node, dsD.get(i, d))
                    d = d + 1

        ds = data.BoxDataSet(dsM, my_list)
        name_map_pd = pd.DataFrame(name_map, columns=column_names)

        return ds, name_map_pd

    def time_knowledge(self, ds, parameters):

        '''
        Create tiers e.g. which variables appear in time lags t-2, t-1 etc
        Parameters
        ----------
            ds(Tetrad object): the dataset in Tetrad format
            parameters (dictionary): the causal configuration
        Returns
        -------
            knowledge(Tetrad object): knowledge about tiers
        '''

        from edu.cmu.tetrad import data

        knowledge = data.Knowledge()
        var_names = list(ds.getVariableNames())

        for t, tier in zip(range(parameters['n_lags'] + 1), reversed(range(parameters['n_lags'] + 1))):

            for i, var in enumerate(var_names):
                if (t == 0) and (':' not in var):
                    knowledge.addToTier(tier, var_names[i])

                elif (t > 0) and (':' + str(t) in var):
                    knowledge.addToTier(tier, var_names[i])

                else:
                    continue

        return knowledge

    def _ci_test(self, ds, parameters):

        '''
        Conditional independence tests in Tetrad
        Args:
            ds(Tetrad object): the dataset in Tetrad format
            parameters (dictionary): the causal configuration
        Returns:
            ind_test(Tetrad object) : the independence test
        '''

        from edu.cmu.tetrad.search import test

        if parameters['ci_test'] == 'FisherZ':
            ind_test = test.IndTestFisherZ(ds, parameters['significance_level'])

        elif parameters['ci_test'] == 'cci':
            ind_test = test.IndTestConditionalCorrelation(ds, parameters['significance_level'])

        elif parameters['ci_test'] == 'cg_lrt':
            discretize = True
            ind_test = test.IndTestConditionalGaussianLrt(ds, parameters['significance_level'], discretize)

        elif parameters['ci_test'] == 'dg_lrt':
            ind_test = test.IndTestDegenerateGaussianLrt(ds)
            ind_test.setAlpha(parameters['significance_level'])

        elif parameters['ci_test'] == 'chisquare':
            ind_test = test.IndTestChiSquare(ds, parameters['significance_level'])

        elif parameters['ci_test'] == 'gsquare':
            ind_test = test.IndTestGSquare(ds, parameters['significance_level'])

        else:
            raise ValueError('%s ind test not included' % parameters['ci_test'])

        return ind_test

    def _score(self, ds, parameters):

        '''
        Scoring functions in Tetrad
        Args:
            ds(Tetrad object): the dataset in Tetrad format
            parameters (dictionary): the causal configuration
        Returns:
            score_(Tetrad object) : the score
        '''

        from edu.cmu.tetrad.search import score

        if parameters['score'] == 'sem_bic_score':
            score_ = score.SemBicScore(ds, True)
            score_.setPenaltyDiscount(parameters['penalty_discount'])

        elif parameters['score'] == 'bdeu':
            score_ = score.BdeuScore(ds)
            score_.setStructurePrior(parameters['structure_prior'])

        elif parameters['score'] == 'discrete_bic':
            score_ = score.DiscreteBicScore(ds)
            score_.setPenaltyDiscount(parameters['penalty_discount'])
            score_.setStructurePrior(parameters['structure_prior'])

        elif parameters['score'] == 'cg_bic':
            discretize = True
            score_ = score.ConditionalGaussianScore(ds, parameters['penalty_discount'], discretize)

        elif parameters['score'] == 'dg_bic':
            score_ = score.DegenerateGaussianScore(ds, True)
            score_.setPenaltyDiscount(parameters['penalty_discount'])
            # score_.setStructurePrior(structure_prior)

        else:
            raise ValueError('%s score not included' % parameters['score'])

        return score_

    def _algo(self, ds, parameters, ind_test, score):

        '''
        Causal discovery algorithms in Tetrad
        Args:
            ds(Tetrad object): the dataset in Tetrad format
            parameters (dictionary): the causal configuration
            ind_test (Tetrad object) : the independence test
            score (Tetrad object) : the score
        Returns:
            alg(Tetrad object) : the causal discovery algorithm
        '''

        from edu.cmu.tetrad import search

        if parameters['name'] == 'pc':
            alg = search.Pc(ind_test)
            alg.setMeekPreventCycles(True)
            alg.setStable(parameters['stable'])
            # alg.setDepth(max_k)
        elif parameters['name'] == 'cpc':
            alg = search.Cpc(ind_test)
            alg.setStable(parameters['stable'])
            alg.meekPreventCycles(True)
            # alg.setDepth(max_k)
        elif parameters['name'] == 'fges':
            alg = search.Fges(score)
        elif parameters['name'] == 'directlingam':
            alg = search.DirectLingam(ds, score)
        elif parameters['name'] == 'fci':
            alg = search.Fci(ind_test)
        elif parameters['name'] == 'fcimax':
            alg = search.FciMax(ind_test)
        elif parameters['name'] == 'rfci':
            alg = search.Rfci(ind_test)
        elif parameters['name'] == 'gfci':
            alg = search.GFci(ind_test, score)
        elif parameters['name'] == 'cfci':
            alg = search.Cfci(ind_test)
            # alg.setDepth(max_k)
        elif parameters['name'] == 'svarfci':
            alg = search.SvarFci(ind_test)
        elif parameters['name'] == 'svargfci':
            alg = search.SvarGfci(ind_test, score)
        else:
            raise ValueError('%s algorithm not included' % parameters['name'])

        return alg

    def output_to_array(self, tetrad_graph_, var_map):

        """Converts tetrad graph to numpy array

            Parameters
            ----------
            tetrad_graph_ (Tetrad object) : graph from tetrad (it can be also a time-lagged graph)
            var_map (pandas Dataframe):  mapping for the variable names

            Returns
            -------
            matrix_pd(pandas Dataframe): matrix of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj   in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j   in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j   in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j
        """

        n_nodes_ = tetrad_graph_.getNumNodes()
        edges = tetrad_graph_.getEdges()
        edgesIterator = edges.iterator()

        matrix = np.zeros(shape=(n_nodes_, n_nodes_), dtype=int)

        while edgesIterator.hasNext():
            curEdge = edgesIterator.next()

            Nodei = str(curEdge.getNode1().toString())
            Nodej = str(curEdge.getNode2().toString())

            iToj = str(curEdge.getEndpoint2().toString())
            jToi = str(curEdge.getEndpoint1().toString())

            i = np.where(var_map['tetrad_name'] == Nodei)
            j = np.where(var_map['tetrad_name'] == Nodej)

            if iToj == 'Circle' or iToj == 'CIRCLE':
                matrix[i, j] = 1
            elif iToj == 'Arrow' or iToj == 'ARROW':
                matrix[i, j] = 2
            elif iToj == 'Tail' or iToj == 'TAIL':
                matrix[i, j] = 3

            if jToi == 'Circle' or jToi == 'CIRCLE':
                matrix[j, i] = 1
            elif jToi == 'Arrow' or jToi == 'ARROW':
                matrix[j, i] = 2
            elif jToi == 'Tail' or jToi == 'TAIL':
                matrix[j, i] = 3


        # for consistency : tail - tail corresponds to o-o
        matrix_t = np.transpose(matrix)
        tail_tail = np.logical_and(matrix == 3, matrix_t == 3)
        matrix[tail_tail] = 1

        matrix_pd = pd.DataFrame(matrix, columns=var_map['var_name'], index=var_map['var_name'])

        return matrix_pd

    def run(self, parameters):

        '''
        Runs the causal discovery configuration
        Parameters
        ----------
            parameters(dictionary): the causal discovery configuration
        Returns
        -------
            library_results(dictionary): save the graphs as returned by Tetrad
            mec_graph_pd(pandas Dataframe) : matrix of the estimated MEC graph
            graph_pd(pandas Dataframe): matrix of the estimated graph


        In any case (temporal or cross sectional data)
            mec_graph_pd and graph_pd are matrices of size N*N where N is the total number of nodes
            and have the following notation:
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs
        '''

        from edu.cmu.tetrad.graph import GraphTransforms
        from edu.cmu.tetrad import data


        ds, var_map = self.prepare_data(parameters)

        if 'ci_test' in parameters.keys():
            ind_test = self._ci_test(ds, parameters)
        else:
            ind_test = None

        if 'score' in parameters.keys():
            score_ = self._score(ds, parameters)
        else:
            score_ = None

        alg = self._algo(ds, parameters, ind_test, score_)

        # set knowledge
        if self.is_time_series:
            tetrad_knowledge = self.time_knowledge(ds, parameters)
            alg.setKnowledge(tetrad_knowledge)

        # prior knowledge
        if isinstance(self.tiers, list) and parameters['name'] != 'directlingam':
            knowledge = data.Knowledge()
            for tier in range(len(self.tiers)):
                for c in range(len(self.tiers[tier])):
                    nodeName = 'X' + str(self.tiers[tier][c] + 1)
                    knowledge.addToTier(tier, nodeName)
            alg.setKnowledge(knowledge)

        # find the MEC graph and/or the DAG or the MAG
        if parameters['name'] != 'directlingam':
            try:
                tetrad_mec_graph = alg.search()
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                return None, None, None

            if parameters['causal_sufficiency']:
                tetrad_graph = GraphTransforms.dagFromCPDAG(tetrad_mec_graph)
            else:
                tetrad_graph = GraphTransforms.pagToMag(tetrad_mec_graph)

        else:
            tetrad_graph = alg.search()
            tetrad_mec_graph = GraphTransforms.cpdagForDag(tetrad_graph)

        mec_graph_pd = self.output_to_array(tetrad_mec_graph, var_map)
        graph_pd = self.output_to_array(tetrad_graph, var_map)

        if parameters['causal_sufficiency']:
            if not is_dag(graph_pd):
             warnings.warn('graph is not a DAG')

        library_results = {}
        library_results['mec'] = tetrad_mec_graph
        library_results['graph'] = tetrad_graph
        library_results['map'] = var_map

        # jpype.shutdownJVM()
        return library_results, mec_graph_pd, graph_pd
