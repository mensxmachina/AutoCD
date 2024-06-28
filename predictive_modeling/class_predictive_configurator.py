
import json
import itertools
import os


class PredictiveConfigurator():
    '''
    Reads the available predictive learning and feature selection algorithms from json files
    and creates the predictive configurations
    Author : kbiza@csd.uoc.gr
    '''

    def __init__(self):

        self.path = os.path.dirname(__file__)
        self.pred_algs = json.load(open(os.path.join(self.path, '../jsons/pred_algs.json')))
        self.fs_algs = json.load(open(os.path.join(self.path,'../jsons/fs_algs.json')))

    def _dict_product(self, dicts):
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


    def create_predictive_configs(self, data_object):

        '''
        Creates list of predictive configurations
        Author: kbiza@csd.uoc.gr
        Returns
        -------
            pred_configs (list of dictionaries) :  the predictive configurations
        '''

        data_type = data_object.data_type
        pred_configs = []
        for pred_name, pred_info in self.pred_algs.items():
            if data_type in pred_info['data_type']:
                # remove data type info from output combinations
                del pred_info['data_type']
                for v in self._dict_product(pred_info):
                    for fs_name, fs_info in self.fs_algs.items():
                        for k in self._dict_product(fs_info):
                            pred_configs.append({"pred_name": pred_name} | v | {"fs_name": fs_name} | k)

        return pred_configs