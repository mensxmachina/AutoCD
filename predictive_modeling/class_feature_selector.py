
import subprocess
import pandas as pd
import os


class Feature_Selector():

    '''
    Feature selection with the MxM R package
    Author: kbiza@csd.uoc.gr, droubo@csd.uoc.gr
    '''

    def __init__(self, data_pd, dataset_name):

        '''
        Parameters
        ----------
            data_pd (pandas Dataframe) : the dataset
            dataset_name (str) :  the name of the csv file with the dataset
        '''

        self.r_path = 'C:/Program Files/R/R-4.3.2/bin/Rscript'   # change this path and/or R version if needed
        self.path_ = os.path.dirname(__file__)
        self.dataset_name = dataset_name + '.csv'
        data_pd.to_csv(os.path.join(self.path_,self.dataset_name), index=False)

    def fbed(self, target_name, config, train_idx_name=None):

        '''
        Runs the FBED feature selection algorithm using the MxM R package
        Parameters
        ----------
            target_name (str) : the name of the target
            config (dictionary) : the predictive configuration
            train_idx_name (str) : the name of the csv file with the train indexes for a specific fold
        Returns
        -------
            selected_features (list): the indexes of the selected features
        '''

        if not train_idx_name:
            subprocess.call([self.r_path, '--vanilla', os.path.join(self.path_, 'fbed_with_idx.R'),
                             self.dataset_name, target_name, config['ind_test_name'], str(config['alpha']), str(config['k'])],
                            shell=True)
        else:
            subprocess.call([self.r_path, '--vanilla', os.path.join(self.path_, 'fbed_with_idx.R'),
                             self.dataset_name, target_name, config['ind_test_name'], str(config['alpha']), str(config['k']), train_idx_name],
                            shell=True)

        selected_features_pd = pd.read_csv(os.path.join(self.path_, 'fbed_selectedVars.csv'))
        if 'sel' in selected_features_pd:
            selected_features = selected_features_pd['sel'].to_list()
            for i in range(len(selected_features)):
                selected_features[i] -= 1
        else:
            selected_features = []

        return selected_features

    def ses(self, target_name, config, train_idx_name=None):

        '''
        Runs the SES feature selection algorithm using the MxM R package
        Parameters
        ----------
            target_name (str) : the name of the target
            config (dictionary) : the predictive configuration
            train_idx_name (str) : the name of the csv file with the train indexes for a specific fold

        Returns
        -------
            selected_features (list): the indexes of the selected features
        '''

        if not train_idx_name:
            subprocess.call([self.r_path, '--vanilla', os.path.join(self.path_, 'ses_with_idx.R'),
                             self.dataset_name, target_name, config['ind_test_name'], str(config['alpha']), str(config['max_k'])],
                            shell=True)
        else:
            subprocess.call([self.r_path, '--vanilla', os.path.join(self.path_, 'ses_with_idx.R'),
                             self.dataset_name, target_name, config['ind_test_name'], str(config['alpha']), str(config['max_k']), train_idx_name],
                            shell=True)

        selected_features_pd = pd.read_csv(os.path.join(self.path_, 'ses_selectedVars.csv'))
        if 'x' in selected_features_pd:
            selected_features = selected_features_pd['x'].to_list()
            for i in range(len(selected_features)):
                selected_features[i] -= 1
        else:
            selected_features = []

        return selected_features


    def feature_selection(self, config, target_name, train_idx_name=None):

        '''
        Feature selection
        Parameters
        ----------
            config (dictionary) : the predictive configuration
            target_name (str) : the name of the target
            train_idx_name (str) : the name of the csv file with the train indexes for a specific fold
        Returns
        -------
            features (list): the indexes of the selected features
        '''

        if config['fs_name'] == 'fbed':
            features = self.fbed(target_name, config, train_idx_name=train_idx_name)

        elif config['fs_name'] == 'ses':
            features = self.ses(target_name, config, train_idx_name=train_idx_name)

        else:
            raise ValueError("not supported feature selection algorithm")

        return features