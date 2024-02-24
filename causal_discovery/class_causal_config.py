
import json
import itertools
import os

class CausalConfigurator():

    '''
    Creates the causal configurations based using the available algorithms from json files
    Author: kbiza@csd.uoc.gr
    '''

    def __init__(self):

        self.path = os.path.dirname(__file__)
        self.causal_algs = json.load(open(os.path.join(self.path, '../jsons/causal_algs.json')))
        self.ci_tests = json.load(open(os.path.join(self.path, '../jsons/ci_tests.json')))
        self.scores = json.load(open(os.path.join(self.path, '../jsons/scores.json')))
        self.n_lags = json.load(open(os.path.join(self.path, '../jsons/lags.json')))

    def _dict_product(self, dicts):
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


    def create_causal_configs(self, data_object, causal_sufficiency):

        '''
        Creates the causal configurations
        Parameters
        ----------
            data_object (class object): contains the data and necessary information
            causal_sufficiency (bool) : True if we need algorithms that assume causal sufficiency

        Returns
        -------
            causal_configs(list of dictionaries) : the causal configurations
        '''

        data_type = data_object.data_type
        is_time_series = data_object.is_time_series

        ci_touse = {}
        for ci_name, ci_info in self.ci_tests.items():
            if data_type in ci_info['data_type']:
                ci_touse[ci_name] = ci_info

        score_touse = {}
        for sc_name, sc_info in self.scores.items():
            if data_type in sc_info['data_type']:
                score_touse[sc_name] = sc_info

        alg_touse = {}
        for alg_name, alg_info in self.causal_algs.items():
            if causal_sufficiency in alg_info['causal_sufficiency']:
                if is_time_series in alg_info['time_series']:
                    alg_touse[alg_name] = alg_info


        causal_configs = []
        for algi_name, algi_info in alg_touse.items():
            for v in self._dict_product(algi_info):

                if is_time_series:
                    for l in self._dict_product(self.n_lags):

                        if 'ci_test' in v.keys() and 'score' not in v.keys():
                            if v['ci_test'] in ci_touse.keys():
                                for k in self._dict_product(ci_touse[v['ci_test']]):
                                    causal_configs.append({"name": algi_name} | v | k | l )

                        elif 'ci_test' not in v.keys() and 'score' in v.keys():
                            if v['score'] in score_touse.keys():
                                for s in self._dict_product(score_touse[v['score']]):
                                    causal_configs.append({"name": algi_name} | v | s | l)

                        elif 'ci_test' in v.keys() and 'score' in v.keys():
                            if v['ci_test'] in ci_touse.keys() and v['score'] in score_touse.keys():
                                for k in self._dict_product(ci_touse[v['ci_test']]):
                                    for s in self._dict_product(score_touse[v['score']]):
                                        causal_configs.append({"name": algi_name} | v | k | s | l)

                else:
                    for l in self._dict_product(self.n_lags):

                        if 'ci_test' in v.keys() and 'score' not in v.keys():
                            if v['ci_test'] in ci_touse.keys():
                                for k in self._dict_product(ci_touse[v['ci_test']]):
                                    causal_configs.append({"name": algi_name} | v | k )

                        elif 'ci_test' not in v.keys() and 'score' in v.keys():
                            if v['score'] in score_touse.keys():
                                for s in self._dict_product(score_touse[v['score']]):
                                    causal_configs.append({"name": algi_name} | v | s )

                        elif 'ci_test' in v.keys() and 'score' in v.keys():
                            if v['ci_test'] in ci_touse.keys() and v['score'] in score_touse.keys():
                                for k in self._dict_product(ci_touse[v['ci_test']]):
                                    for s in self._dict_product(score_touse[v['score']]):
                                        causal_configs.append({"name": algi_name} | v | k | s )

        return causal_configs