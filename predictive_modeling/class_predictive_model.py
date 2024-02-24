
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

class Predictive_Model():

    '''
    Predictive modeling using sklearn
    Author: kbiza@csd.uoc.gr, droubo@csd.uoc.gr
    '''

    def random_forest(self, config, target_type):

        '''
        Parameters
        ----------
            config(dictionary): the predictive configuration
            target_type(str): continuous or categorical

        Returns
        -------
            model(sklearn model): the predictive model
        '''

        if target_type == 'categorical':

            model = RandomForestClassifier(n_estimators=int(config['n_estimators']),
                                           min_samples_leaf=config['min_samples_leaf'],
                                           max_features=config['max_features'])
        else:

            model = RandomForestRegressor(n_estimators=int(config['n_estimators']),
                                          min_samples_leaf=config['min_samples_leaf'],
                                          max_features=config['max_features'])

        return model

    def linear_regression(self):

        model = LinearRegression()
        return model


    def predictive_modeling(self,config,target_type, train_X, train_y, test_X=None):

        '''

        Parameters
        ----------
            config(dictionary): the predictive configuration
            target_type(str): continuous or categorical
            train_X(numpy array): the train samples of the features
            train_y(numpy array): the train samples of the target
            test_X(numpy array or None) : the test samples of the features

        Returns
        -------
            model(sklearn model) : the predictive model
            predictions(numpy array or None): the predictions if test_X is given
            predict_probs(numpy array or None): the predicted probabilities by the classifier if test_X is given
                                                and the target is categorical variable
        '''

        if config['pred_name'] == 'random_forest':
            model = self.random_forest(config, target_type)

        elif config['pred_name'] == 'linear_regression':
            model = self.linear_regression()

        else:
            raise ValueError("not supported predictive algorithm")

        train_y = train_y.reshape(-1,)
        model.fit(train_X, train_y)
        if test_X is None:
            predictions = None
            predict_probs = None
        else:
            predictions = model.predict(test_X)
            if target_type=='categorical':
                predict_probs = model.predict_proba(test_X)
            else:
                predict_probs = None

        return model, predictions, predict_probs

