
import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI

from AutoCD.causal_graph_utils.cpdag_to_dag import *
from AutoCD.causal_graph_utils.pag_to_mag import *
from AutoCD.causal_graph_utils.is_dag import *
import warnings

class Tigramite():

    '''
    Use the Tigramite package to run causal discovery algorithms
    Author: kbiza@csd.uoc.gr
    '''

    def __init__(self, data_pd, var_names_lagged, data_type_info):
        '''
        Parameters
        ----------
            data_pd (pandas Dataframe): the dataset
            var_names_lagged (list) : the names of the variables in each time lag
            data_type_info (pandas Dataframe) : information about the type of each variable (continuous or categorical)
        '''

        self.data_pd = data_pd
        self.data_type_info = data_type_info
        self.var_names = var_names_lagged

    def prepare_data(self, parameters):

        '''
        Converts the input dataset to Tigramite format
        Parameters
        ----------
            parameters(dictionary): the causal configuration parameters
        Returns
        -------
            dataframe_(Tigramite object): the dataset
        '''

        if parameters['ci_test'] != 'RegressionCI':
            dataframe_ = pp.DataFrame(self.data_pd.to_numpy(),
                                 var_names=self.data_pd.columns)
        else:
            data_type = np.zeros(self.data_pd.shape, dtype='int')
            data_type_ = self.data_type_info['var_type'].to_numpy().copy()
            data_type_ = data_type_[0:self.data_pd.shape[1]]  # take only first lag
            data_type_[data_type_ == 'continuous'] = 0
            data_type_[data_type_ == 'categorical'] = 1
            data_type[:] = data_type_

            dataframe_ = pp.DataFrame(self.data_pd.to_numpy(),
                                     data_type=data_type,
                                     var_names=self.data_pd.columns)

        return dataframe_

    def _ci_test(self, parameters):

        '''
        Conditional independence tests in Tigramite
        Args:
            parameters (dictionary): the causal configuration
        Returns:
            ind_test(Tigramite object) : the independence test
        '''

        if parameters['ci_test'] == 'ParCor':  # (significance='analytic')
            ci_test = ParCorr()
        elif parameters['ci_test'] == 'RobustParCor':
            ci_test = RobustParCorr()
        # elif parameters['ci_test'] == 'GPDC':
        #     ci_test = GPDC(significance='analytic', gp_params=None)
        elif parameters['ci_test'] == 'CMIknn':
            ci_test = CMIknn(significance='fixed_thres', model_selection_folds=3)
        elif parameters['ci_test'] == 'ParCorrWLS':
            ci_test = ParCorrWLS(significance='analytic')
        elif parameters['ci_test'] == 'Gsquared':  # for discrete variables
            ci_test = Gsquared(significance='analytic')
        elif parameters['ci_test'] == 'CMIsymb':
            ci_test = CMIsymb(significance='shuffle_test')
        elif parameters['ci_test'] == 'RegressionCI':
            ci_test = RegressionCI(significance='analytic')
        else:
            raise ValueError('%s ci test not included' % parameters['ci_test'])

        return ci_test

    def _algo (self, dataframe_, parameters, ci_test):

        '''
        Causal discovery algorithms in Tigramite
        Args:
            dataframe_(Tigramite object): the dataset
            parameters (dictionary): the causal configuration
            ci_test (Tigramite object) : the independence test
        Returns:
            output(Tigramite object) : the causal discovery algorithm
        '''

        if parameters['name'] == 'PCMCI':
            alg = PCMCI(
                dataframe=dataframe_,
                cond_ind_test=ci_test,
                verbosity=0)
            output = alg.run_pcmci(tau_max=parameters['n_lags'], pc_alpha=parameters['significance_level'],
                                   alpha_level=parameters['significance_level'])

        elif parameters['name'] == 'PCMCI+':
            alg = PCMCI(
                dataframe=dataframe_,
                cond_ind_test=ci_test,
                verbosity=0)
            output = alg.run_pcmciplus(tau_max=parameters['n_lags'], pc_alpha=parameters['significance_level'])

        elif parameters['name'] == 'LPCMCI':
            alg = LPCMCI(
                dataframe=dataframe_,
                cond_ind_test=ci_test,
                verbosity=0)
            output = alg.run_lpcmci(tau_max=parameters['n_lags'], pc_alpha=parameters['significance_level'])

        else:
            raise ValueError('%s cd alg not included' % parameters['name'])

        return output

    def output_to_array(self, output):

        """Converts tigramite graph to matrix of time-lagged causal graph
                i.e. every variable appears in all time lags

            Parameters
            ----------
            output (Tigramite object) : output graph from tigramite (time series graph)

            Returns
            -------
            matrix_pd(pandas Dataframe): matrix of size N*N where N is the number of nodes over all time lags
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj   in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j   in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j   in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j
        """

        t_graph = output['graph']
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

        matrix_pd = pd.DataFrame(matrix, columns=self.var_names, index=self.var_names)

        return matrix_pd

    def run(self, parameters):

        '''
        Runs the causal discovery configuration
        Parameters
        ----------
            parameters(dictionary): the causal discovery configuration
        Returns
        -------
            library_results(dictionary): save the graphs as returned by Tigramite
            mec_graph_pd(pandas Dataframe) : matrix of the estimated MEC graph
            graph_pd(pandas Dataframe): matrix of the estimated graph

            mec_graph_pd and graph_pd are matrices of time-lagged causal graphs
            -of size N*N where N is the total number of nodes over all time lags
            -with the following notation:
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs
        '''

        dataframe_ = self.prepare_data(parameters)
        # print('running tigramite')
        ci_test = self._ci_test(parameters)
        output = self._algo(dataframe_, parameters, ci_test)
        # print('end of running tigramite')
        mec_graph_pd = self.output_to_array(output)

        if parameters['causal_sufficiency']:
            graph_pd = cpdag_to_dag(mec_graph_pd, False, n_lags=parameters['n_lags'])
            if not is_dag(graph_pd):
                warnings.warn('graph is not a DAG')
        else:
            graph_pd = pag_to_mag(mec_graph_pd, False, n_lags=parameters['n_lags'])

        # In this work, graph_pd is only used for markov boundary discovery

        library_results = {}
        library_results['mec'] = output


        return library_results, mec_graph_pd, graph_pd

