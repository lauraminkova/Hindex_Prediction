# ------------------------------------------------------------------- #
#     Script that will run our regression pipeline from end to end    #
# ------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import random
import pickle
import os
import networkx as nx
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import  RobustScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from regressors.cross_validation import cv

random.seed(123)

def full_pipeline(train, test, regressor, params, outfile) :
    '''
    train:          (Dataframe) DataFrame with training authors and all of their features, including target variable h-index
    test:           (DataFrame) DataFrame with test authors and all of their features
    regressor:      (str)       Can be one of sklearn's LogisticRegression, Lasso, Ridge, KNN, MLP, etc.
    params:         (dict)      Dictionary with regressor's parameters 
    outfile:        (str)       Path to where you want to save your predictions
    '''

    # Loading the test submission file
    test_submission = pd.read_csv('test.csv', dtype={'author': np.int64})

    y_train = train.hindex
    train.drop('hindex', axis=1, inplace=True)

    # Will 10-fold cross validation to ensure that your model is not overfitting
    # Will print the scores and average score for this model
    cv(train, y_train, regressor, params)

    if regressor == MLPRegressor :
        norm = Normalizer().fit(train)
        train = norm.transform(train)
        test = norm.transform(test)

    scaler =  RobustScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    reg = regressor(**params)

    reg.fit(train, y_train)
    y_pred = reg.predict(test)

    # If predicted value is negative, change to 0.
    for i in range(len(y_pred)):
        if y_pred[i] < 0 :
            y_pred[i] = 0

    # write the predictions to file
    test_submission['hindex'] = pd.Series(np.round_(y_pred, decimals=3))
    test_submission.loc[:,["author","hindex"]].to_csv(outfile, index=False)


########### EXAMPLE RUN ###########
###################################

# Make sure these paths are correct before you run
train = pd.read_parquet("final_train_df_whindex.parquet")
test = pd.read_parquet("final_test_df.parquet")

# Change this to whatever regressor your want
regressor = MLPRegressor
params = {'activation': 'logistic',
        'alpha': 0.001,
        'early_stopping': True,
        'hidden_layer_sizes': 500,
        'learning_rate': 'invscaling',
        'learning_rate_init': 0.0001,
        'verbose': True}

# Change for what ever path you want
outfile = "mysubmission.csv"

full_pipeline(train, test, regressor, params, outfile)