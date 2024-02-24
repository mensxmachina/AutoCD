
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
import pandas as pd
import numpy as np
import os


class OOS():

    '''
    Out-of-sample protocols for data splitting
    Author: kbiza@csd.uoc.gr, droubo@csd.uoc.gr
    '''

    def data_split(self, oos_protocol, X, y=None, target_type='continuous'):

        '''

        Parameters
        ----------
            oos_protocol(dictionary) : information about the out-of-sample protocol
            X (pandas or numpy array) :  the feature data, if y==None, X contains all samples
            y (pandas or numpy array) : the target vector in case of categorical target (for stratified kfold)
            target_type (str) : continuous or categorical

        Returns
        -------
            train_inds (list of lists) : the train indexes
            test_inds(list of lists) : the test indexes

        '''

        train_inds = []
        test_inds = []

        if oos_protocol['name'] == 'KFoldCV':

            # if time_series_split:
            #     kf = TimeSeriesSplit(n_splits=oos_protocol['folds'])
            # else:
            #     kf = KFold(n_splits=oos_protocol['folds'])

            if target_type == 'continuous':
                kf = KFold(n_splits=oos_protocol['folds'])
                kf_split = kf.split(X)
            else:
                kf = StratifiedKFold(n_splits=oos_protocol['folds'])
                kf_split = kf.split(X, y)

            for i, (train_index, test_index) in enumerate(kf_split):
                train_inds.append(train_index)
                test_inds.append(test_index)
                # save train index for R
                csv_path = os.path.dirname(__file__)
                pd.DataFrame(train_index, columns=["train_idx"]).to_csv(os.path.join(csv_path,"train_idx_" + str(i) + ".csv"), index=False)

        else:
            raise ValueError('%s protocol not included' % oos_protocol['name'])

        return train_inds, test_inds
