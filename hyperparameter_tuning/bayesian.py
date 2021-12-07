# ------------------------------------------------------- #
#     Script for Bayesian hyperparameter optimization     #
# ------------------------------------------------------- #

import os
import sys
import pandas as pd
import numpy as np
import random
# import ast
from datetime import datetime

os.chdir("/home/laura/INF554-Final-Project")

from hyperopt import tpe, fmin, Trials, STATUS_OK, hp
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from statistics import mean
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Logistic regression
bayes_lr = {'penalty': hp.choice('penalty', ['l1', 'l2']),
           'C': hp.choice('C', [0.001, 0.01, 0.1, 1, 10]),
           'class_weight': hp.choice('class_weight', [None, 'balanced']),
           'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
           'max_iter': 1000,
           'random_state': 123}

# LASSO
bayes_lsso = {'alpha': hp.choice('alpha', [1, 0.1, 0.01, 0.001]),
             'selection': hp.choice('selection', ['cyclic', 'random']),
             'random_state' : 123}

# Ridge
bayes_ridg = {'alpha': hp.choice('alpha', [1, 0.1, 0.01, 0.001]),
             'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga', 'lbfgs', 'sag']),
             'random_state': 123}

# SVR
bayes_svr = {'C': hp.choice('C', [1, 10, 100, 1000]),
            'gamma': hp.choice('gamma', ['scale', 'auto']),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
            'max_iter': 100000}

# Random Forest': 
bayes_rf = {'n_estimators': scope.int(hp.quniform('n_estimators', 100, 400, 1)),
            'criterion': hp.choice('criterion', ['squared_error', 'absolute_error', 'poisson']),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            'random_state': 123,
            'n_jobs': -1}

# K-Nearest Neighbors
bayes_knn = {'n_neighbors': hp.choice('n_neighbors', [3, 6, 12, 18]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'metric': hp.choice('metric', ['manhattan', 'chebyshev', 'minkowski']),
            'n_jobs': -1}

bayes_mlp = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [100, 200, 500]),
            'activation': hp.choice('activation', ['relu', 'identity', 'logistic', 'tanh']),
            'alpha': hp.choice('alpha', [0.01, 0.001, 0.001]), 
            'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']), 
            'learning_rate_init':hp.choice('learning_rate_init', [0.01, 0.001, 0.0001]), 
            'verbose' : True, 
            'early_stopping' : True}

MODELS_DICT = {'LR': {'model': LogisticRegression, 'bayes_params': bayes_lr}, 
              'LASSO': {'model': Lasso, 'bayes_params': bayes_lsso},
              'Ridge': {'model': Ridge, 'bayes_params': bayes_ridg},
              'SVR': {'model': SVR, 'bayes_params': bayes_svr},
              'RF': {'model': RandomForestRegressor, 'bayes_params': bayes_rf},
              'KNN': {'model': KNeighborsRegressor, 'bayes_params': bayes_knn},
              'MLP': {'model': MLPRegressor, 'bayes_params': bayes_mlp},}



def bayes_hopt(name, train, y_train, regressor, normalize=False) :
    '''
    name:       (str) Name you would give this run
    training:   (str) Path to training file
    regressor:  (str) Short form of regressor you want to use
    '''

    # Checking if we've already optimized hyperparameters for this specific exp
    file = 'params/{}_params.txt'.format(name)
    if os.path.isfile(file):
        best_parameters = get_model_params(file)
        
    else:                     
        
        ####### Prepping training data #######
        if normalize :
            norm = Normalizer().fit(train)
            train = norm.transform(train)

        scaler =  RobustScaler().fit(train)
        train = scaler.transform(train)

        #######################################

        model = MODELS_DICT[regressor]['model']
        search_space = MODELS_DICT[regressor]['bayes_params']

        trials = Trials()

        def objective_function_datasets(search_space): 
            
            reg = model(**search_space)
            scores = cross_val_score(reg, train, y_train, cv=10, scoring='neg_mean_squared_error')
            scores = -1*scores
            final_score = mean(scores)

            return {'loss': final_score, 'status' : STATUS_OK}

        best_parameters = fmin(objective_function_datasets, search_space, algo=tpe.suggest, max_evals=20, trials=trials, 
                               rstate=np.random.RandomState(123), return_argmin=False, early_stop_fn=no_progress_loss(iteration_stop_count=50, percent_increase=0.0))

                  
        # Storing all the tuned hyperparameters to avoid waiting to tune every time
        file = open('params/{}_params.txt'.format(name), "wb")
        file.write(str(best_parameters))
        file.close()

        #Saving trials
        pickle.dump(trials, open("params/trials/{}".format(name), "wb"))

    return best_parameters

def get_model_params(file):
    '''
    Method that looks for the 'model' parameters that were
    generated by hyperparameter optimization
    '''

    parameter_file = open(file, "r")
    params = parameter_file.readline()
    params = ast.literal_eval(params)
    parameter_file.close()

    return params

if __name__ == "__main__" :

    s = datetime.now()

    train = pd.read_parquet('data/train-graph-text-bert-prone-node2vec-DBSCAN-071221.parquet')
    y_train = train['hindex']
    train.drop('hindex', axis=1, inplace=True)

    params = bayes_hopt('MLP-most-recent', train, y_train, 'MLP', normalize=False)
    print(params)

    print(datetime.now() - s)
    