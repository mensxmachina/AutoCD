
import subprocess
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, roc_auc_score
from AutoCD.predictive_modeling.class_feature_selector import *
from AutoCD.predictive_modeling.class_predictive_model import *
from AutoCD.predictive_modeling.class_oos import *
from AutoCD.data_utils.functions_data import *
import os


class AFS():

    '''
    The AFS module of the AutoCD architecture
    Author: kbiza@csd.uoc.gr, droubo@csd.uoc.gr
    '''

    def __init__(self):

        self.oos_protocol={"name": "KFoldCV",
                            "folds": 5,
                            "folds_to_run": 5}

        self.csv_path = os.path.dirname(__file__)

    def run_AFS(self, pred_configs, data_object):

        '''
        Runs the AFS module
        Parameters
        ----------
            pred_configs (list of dictionaries) :  the predictive configurations
            data_object (class object) : the data object which contains the dataset and other necessary information

        Returns
        -------
            mb_idx (list) : indexes of the selected features (markov boundary) of the target node
            mb_names (list) : names of the selected features (markov boundary) of the target node
            avg_performance (float): the average out-of-sample predictive performance of the selected model
            best_model (sklearn model): the selected predictive model
            best_config (dictionary): the selected predictive configuration
        '''

        if data_object.is_time_series:
            # run analysis on time lagged dataset
            data_pd = timeseries_to_timelagged(data_object.samples, data_object.nlags)
        else:
            data_pd = data_object.samples

        target_info = data_object.target_info
        dataset_name = data_object.dataset_name
        target_name = target_info['name']
        target_type = target_info['var_type']

        fs = Feature_Selector(data_pd, dataset_name)
        pm = Predictive_Model()

        # Cross-validation to find optimal configuration
        n_nodes = data_pd.shape[1]
        np_data = data_pd.to_numpy()

        target_idx = np.where(data_pd.columns == target_name)[0]
        x_idx = [x for x in range(n_nodes) if x != target_idx]
        X = np_data[:, x_idx].copy()
        y = np_data[:, target_idx].copy()

        train_inds, test_inds = OOS().data_split(self.oos_protocol, X, y, target_type)

        scores_ = np.empty((len(pred_configs), len(train_inds)), dtype=float)

        for c, config in zip(range(len(pred_configs)), pred_configs):

            for i, (train_idx, test_idx) in enumerate(zip(train_inds, test_inds)):

                train_X, test_X = X[train_idx].copy(), X[test_idx].copy()
                train_y, test_y = y[train_idx].copy(), y[test_idx].copy()

                # Feature selection
                # os.path.join(self.csv_path, self.dataset_name)
                train_idx_name = 'train_idx_' + str(i) + '.csv'
                features = fs.feature_selection(config, target_name, train_idx_name=train_idx_name)

                # Predictive modeling
                if features:
                    train_X_sel = train_X[:, features].copy()
                    test_X_sel = test_X[:, features].copy()

                    model, predictions, predict_probs = pm.predictive_modeling(config, target_type, train_X_sel, train_y, test_X_sel)
                else:
                    if target_type == 'continuous':
                        predictions = np.ones(test_y.shape[0]) * np.mean(train_y)
                    else:
                        values, counts = np.unique(test_y, return_counts=True)
                        predictions = np.full(test_y.shape[0], values[np.argmax(counts)])
                        predict_probs = np.full((test_y.shape[0],len(values)), 0, dtype=float)
                        predict_probs[:,np.argmax(counts)]=1


                if target_type == 'continuous':
                    scores_[c, i] = r2_score(test_y, predictions)
                else:
                    test_y_ = test_y.reshape(-1,)
                    scores_[c, i] = roc_auc_score(test_y_, predict_probs, multi_class='ovr')


        # Select configuration
        avg_fold_perf = np.mean(scores_, axis=1)
        idx_best_configs = np.argmax(avg_fold_perf, axis=0)
        best_config = pred_configs[idx_best_configs]
        print('target', target_name, 'with optimal config:', best_config)

        # Apply selected configuration on all data
        # Feature selection
        features = fs.feature_selection(best_config, target_name, train_idx_name=None)

        # Predictive modeling
        if features:
            X_sel = X[:, features].copy()
            best_model, predictions_, _ = pm.predictive_modeling(best_config, target_type, X_sel, y)
            mb = np.asarray(x_idx)[features]
            mb = mb.tolist()
        else:
            best_model = []
            mb = []

        mb_idx = mb
        mb_names = data_pd.columns[mb_idx].to_list()
        avg_performance = np.max(avg_fold_perf)

        return mb_idx, mb_names, avg_performance, best_model, best_config


