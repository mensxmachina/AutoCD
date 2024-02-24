from sklearn.metrics import mutual_info_score, roc_auc_score
from statistics import stdev
from math import log, pi, exp
import json
import copy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from AutoCD.causal_graph_utils.markov_boundary import *
from AutoCD.causal_discovery.functions_causal_discovery import *
from AutoCD.predictive_modeling.class_oos import *
from AutoCD.data_utils.functions_data import *

class OCT_parallel():

    '''
    The OCT tuning method of the CL module of the AutoCD architecture
    Authors: kbiza@csd.uoc.gr, droubo@csd.uoc.gr
    '''

    def __init__(self, dataObject,  n_jobs):

        '''

        Parameters
        ----------
            dataObject (class object) : contains the data and relevant information
            n_jobs (int) : number of jobs for parallel computation
        '''

        self.path = os.path.dirname(__file__)
        self.oct_params = json.load(open(os.path.join(self.path, '../jsons/oct_params.json')))
        self.oos_protocol = self.oct_params['out_of_sample_protocol']
        self.regres_params = self.oct_params['regressor']
        self.classif_params = self.oct_params['classifier']
        self.sparsity_params = self.oct_params['sparsity_penalty']

        self.saved_mb_configs = {}
        self.saved_pred_configs = {}
        self.saved_y_test = {}
        self.saved_mb_size = {}

        self.samples = dataObject.samples
        self.is_time_series = dataObject.is_time_series
        self.data_type_info = dataObject.data_type_info

        self.n_jobs=n_jobs


    def mutual_info_continuous(self, y, y_hat):

        """
        Computes the mutual information between two continuous variables, assuming Gaussian distribution
        Args:
            y (numpy array): vector of true values
            y_hat (numpy array): vector of predicted values

        Returns:
            mutual_info (float) : mutual information of y and y_hat
        """

        if stdev(y) == 0 or stdev(y_hat) == 0:
            raise ValueError("MutualInfo: zero st_dev")

        std_y = stdev(y)
        std_y_hat = stdev(y_hat)

        h_y = (1/2) * log(2 * pi * exp(1) * (std_y ** 2))
        h_y_hat = (1/2) * log(2 * pi * exp(1) * (std_y_hat ** 2))

        if np.array_equal(y, y_hat):
            mutual_info = h_y
        else:
            corr = np.corrcoef(y, y_hat)[0,1]
            mutual_info = -(1/2)*log(1-corr**2)

        return mutual_info


    def fold_fit(self, target, c, graphs_configs, train_inds, test_inds, fold):

        """
        Performs Markov boundary identification of the target and predictive modeling
        for a single fold and a particular configuration.
        Author : droubo@csd.uoc.gr

        Parameters:
            target (int): Target index.
            c (int): Configuration index.
            mec_graphs_configs (list of numpy arrays): MEC graphs configurations.
            data_train (pandas DataFrame): Training data.
            data_test (pandas DataFrame): Testing data.
            fold (int): Fold index.

        Returns:
            mb (numpy array): Markov boundary.
            prediction (numpy array): Predicted values.
            y_test (numpy array): Actual target values for the test data.
        """

        mb = markov_boundary(target, graphs_configs[c][fold])  # if mag
        len_mb = len(mb)

        train_samples_ = self.samples.iloc[train_inds[fold]]
        test_samples_ = self.samples.iloc[test_inds[fold]]

        if self.is_time_series:
            train_samples = timeseries_to_timelagged(train_samples_, self.n_lags_config)
            test_samples = timeseries_to_timelagged(test_samples_, self.n_lags_config)
        else:
            train_samples = train_samples_
            test_samples = test_samples_

        X_train = train_samples.iloc[:, mb]
        y_train = train_samples.iloc[:, target]
        X_test = test_samples.iloc[:, mb]
        y_test = test_samples.iloc[:, target]

        if self.data_type_info_['var_type'].iloc[target] == 'categorical':
            if len(mb) > 0:
                clf = copy.deepcopy(RandomForestClassifier(n_estimators=self.classif_params['n_trees'], random_state=0))
                clf.fit(X_train, y_train)
                prediction = clf.predict(X_test)
                # predict_probs = clf.predict_proba(X_test)
                # y_test_ = y_test.reshape(-1, )
                # score_ = roc_auc_score(y_test_, predict_probs, multi_class='ovr')
                # score_ = clf.score(X_test, y_test)
            else:
                values, counts = np.unique(y_test, return_counts=True)
                prediction = np.full(y_test.shape[0], values[np.argmax(counts)])
                # predict_probs = np.full((y_test.shape[0], len(values)), 0, dtype=float)
                # predict_probs[:, np.argmax(counts)] = 1

        else:
            if len(mb) > 0:
                clf = copy.deepcopy(RandomForestRegressor(n_estimators=self.regres_params['n_trees'], random_state=0))
                clf.fit(X_train, y_train)
                prediction = clf.predict(X_test)
                # score_ = clf.score(X_test, y_test)
            else:
                prediction = np.full(y_test.shape[0], np.mean(y_train))

        return [mb, len_mb, prediction, y_test]


    def nodes_parallel(self, target, c, graphs_configs, train_indexes, test_indexes):

        """
        Calculates the mutual information between the true values and
        predicted values of a target node
        Author: droubo@csd.uoc.gr

        Parameters:
            target (int): target node id
            c (int): regularization parameter
            mec_graphs_configs (list): list of mec graph configurations
            data_train (numpy array): training data
            data_test (numpy array): test data

        Returns:
            mu (float): mutual information score between the true values and predicted values
            mb_folds (list): list of Markov blanket of each fold
            pred_folds (list): list of predictions for each fold
            y_test_folds (list): list of true values for each fold

        """

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fold_fit)(target, c, graphs_configs, train_indexes, test_indexes, fold)
            for fold in range(self.oos_protocol['folds']))

        results = np.array(results, dtype=object)
        mb_folds = results[:, 0]
        len_mb_folds = results[:,1]
        pred_folds = results[:, 2]
        y_test_folds = results[:, 3]

        # Predictive performance for pooled out-of-sample predictions
        pred_folds_np = np.concatenate(pred_folds, axis=0)
        y_test_folds_np = np.concatenate(y_test_folds, axis=0)

        if self.data_type_info_['var_type'].iloc[target] == 'categorical':
            mu = mutual_info_score(y_test_folds_np, pred_folds_np)
        else:
            mu = self.mutual_info_continuous(y_test_folds_np, pred_folds_np)

        return [mu, mb_folds, len_mb_folds,  pred_folds, y_test_folds]


    def config_parallel(self, c, causal_configs, train_indexes, test_indexes):

        """
        Calculates the mutual information scores for all target nodes in a parallel manner
        Author : droubo@csd.uoc.gr

        Parameters:
            c (int): regularization parameter
            mec_graphs_configs (list): list of mec graph configurations
            data_train (numpy array): training data
            data_test (numpy array): test data

        Returns:
            mu (list): list of mutual information scores for each target node
            mb_folds (list): list of Markov blanket of each fold for each target node
            pred_folds (list): list of predictions for each fold for each target node
            y_test_folds (list): list of true values for each fold for each target node

        """

        print('run config parallel', c)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.nodes_parallel)(target, c, causal_configs, train_indexes, test_indexes)
            for target in range(self.n_nodes))

        results = np.array(results, dtype=object)

        return [results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4]]


    def _sparsity_penalty(self,  opt_config):

        '''
        Performs OCT's sparsity penalty
        Parameters
        ----------
            opt_config (dictionary) : the selected causal configuration

        Returns
        -------
            OCTs_c (int) : the index of the new causal configuration
        '''

        print('\tSparsity penalty')

        # config --> node --> fold
        n_configs = self.pred_configs.shape[0]

        # if time series we choose
        #   the minimum sample size in time-lagged data and  this does not affect cross-sectional analysis
        n_samples_lagged = []
        for node in range(len(self.y_test_nodes)):
            for fold in range(len(self.y_test_nodes[node])):
                n_samples_lagged.append(len(self.y_test_nodes[node][fold]))
        n_samples_min_lagged = np.min(n_samples_lagged)
        n_samples_study = n_samples_min_lagged * self.oos_protocol['folds_to_run']

        # indexes for permutation
        idxs = []
        for i in range(self.sparsity_params['n_permutations']):
            idx_bool = np.random.randint(0, high=2, size=n_samples_study, dtype=bool)
            idx = np.where(idx_bool)[0]
            idxs.append(idx)

        # permutation test
        p_values = np.ones((n_configs))
        is_equal = np.ones((n_configs), dtype=bool)

        mean_mb=[]
        for c in range(n_configs):
            avg_folds = []
            for n in range(self.n_nodes):
                avg_mb_fold = np.mean(self.len_mb_configs[c,n]) #avg over folds
                avg_folds.append(avg_mb_fold)
            mean_mb.append(np.mean(avg_folds))

        for c in range(n_configs):

            if c == opt_config:
                continue

            swap_best_metric = np.zeros((self.sparsity_params['n_permutations'], self.n_nodes))
            swap_cur_metric = np.zeros((self.sparsity_params['n_permutations'], self.n_nodes))

            for node in range(self.n_nodes):
                poolYhat_best = self.pred_configs[opt_config,node]
                poolYhat_best = np.concatenate([item[0:n_samples_min_lagged] for item in poolYhat_best], axis=0)

                poolYhat_cur = self.pred_configs[c,node]
                poolYhat_cur = np.concatenate([item[0:n_samples_min_lagged] for item in poolYhat_cur], axis=0)

                poolY = self.y_test_nodes[node]
                poolY = np.concatenate([item[0:n_samples_min_lagged] for item in poolY], axis=0)


                # swap predictions
                for i in range(self.sparsity_params['n_permutations']):

                    idx = idxs[i]

                    swap_best = poolYhat_best.copy()
                    swap_best[idx] = poolYhat_cur[idx]

                    swap_cur = poolYhat_cur.copy()
                    swap_cur[idx] = poolYhat_best[idx]

                    if self.data_type_info_['var_type'].iloc[node] == 'categorical':
                        swap_best_metric[i, node] = mutual_info_score(poolY, swap_best)
                        swap_cur_metric[i, node] = mutual_info_score(poolY, swap_cur)
                    else:
                        swap_best_metric[i, node] = self.mutual_info_continuous(poolY, swap_best)
                        swap_cur_metric[i, node] = self.mutual_info_continuous(poolY, swap_cur)

            curMetric = self.mean_mu_configs[c]
            bestMetric = np.max(self.mean_mu_configs)
            obs_t_stat = bestMetric - curMetric
            t_stat = np.mean(swap_best_metric, axis=1) - np.mean(swap_cur_metric, axis=1)

            p_val = np.count_nonzero(t_stat >= obs_t_stat) / self.sparsity_params['n_permutations']
            p_values[c] = p_val

            # H0: the difference in performance is zero
            if p_val > self.sparsity_params['alpha']:
                is_equal[c] = True
            else:
                is_equal[c] = False

        # select configuration
        OCTs_c = opt_config
        for c in range(n_configs):
            if np.logical_and(is_equal[c], mean_mb[c] < mean_mb[OCTs_c]):
                OCTs_c = c

        return OCTs_c


    def select(self, causal_configs):

        '''
        Runs the OCT tuning method
        Parameters
        ----------
            causal_configs (list of dictionaries) :  the causal configurations to choose from

        Returns
        -------
            library_results (#dictionary): the causal results as returned from the corresponding CD package
            mec_graph_pd (pandas Dataframe) : the estimated Markov Equivalence Class (PDAG or PAG)
            graph_pd (pandas Dataframe) : a causal graph (DAG or MAG) from the estimated MEC
            optimal_config (dictionary) : the selected causal configuration
        '''

        print('OCT tuning method')

        if self.is_time_series:
            self.n_lags_config = causal_configs[0]['n_lags']
            self.n_nodes = self.samples.shape[1] * (self.n_lags_config + 1)
            var_names_lagged = []
            for lag in range(self.n_lags_config + 1):
                if lag == 0:
                    var_names_lagged.extend(self.samples.columns)
                else:
                    var_names_lagged.extend(self.samples.columns + ':' + str(lag))
            self.data_type_info_ = pd.DataFrame(np.tile(self.data_type_info.to_numpy(), [self.n_lags_config + 1, 1]),
                                           index=var_names_lagged, columns=self.data_type_info.columns)
        else:
            self.n_nodes = self.samples.shape[1]
            self.data_type_info_ = self.data_type_info


        # Data partition for OCT
        train_inds, test_inds = OOS().data_split(self.oos_protocol, self.samples)

        # Causal discovery
        mec_graphs_configs = []
        graphs_configs = []

        for config in causal_configs:
            mec_graphs_folds = []
            graphs_folds = []

            for fold in range(self.oos_protocol['folds_to_run']):
                library_results, mec_graph_pd, graph_pd = \
                    causal_discovery(config, self.samples.iloc[train_inds[fold]], self.data_type_info, self.is_time_series)

                mec_graphs_folds.append(mec_graph_pd)
                graphs_folds.append(graph_pd)

            mec_graphs_configs.append(mec_graphs_folds)
            graphs_configs.append(graphs_folds)

        # Parallel
        print('\tPredictive modeling')
        self.mb_size = np.zeros(
            (len(causal_configs), self.oos_protocol['folds_to_run'], self.n_nodes))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.config_parallel)
            (c, graphs_configs, train_inds, test_inds)
            for c in range(len(causal_configs)))

        results = np.array(results, dtype=object)
        # config -> node -> fold
        self.mu_configs = results[:, 0]
        self.mb_configs = results[:, 1]
        self.len_mb_configs = results[:, 2]
        self.pred_configs = results[:, 3]
        self.y_test_nodes = results[:, 4][0]

        mu_configs_np = np.stack(self.mu_configs, axis=0)
        self.mean_mu_configs = np.nanmean(mu_configs_np, axis=1)
        opt_config = np.argmax(self.mean_mu_configs)

        # Sparsity penalty
        OCTs_c = self._sparsity_penalty(opt_config)
        optimal_config = causal_configs[OCTs_c]
        print('Selected causal configuration by OCT', optimal_config)

        # Causal discovery with optimal configuration
        library_results, mec_graph_pd, graph_pd = \
            causal_discovery(optimal_config, self.samples, self.data_type_info, self.is_time_series)

        return library_results, mec_graph_pd, graph_pd, optimal_config