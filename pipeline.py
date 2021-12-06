# ------------------------------------------------------------------- #
#  Script that will run our classification pipeline from end to end   #
# ------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import random
import pickle
import os
import networkx as nx
import classifiers.classify as cl
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler,  PowerTransformer, Normalizer
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

os.chdir("/home/laura/Documents/Polytechnique/MScT - M1/INF554 Machine Learning/Kaggle Data Challenge/INF554-Final-Project")

random.seed(123)

def full_pipeline(graph_data, abs_data, paper_data, ls_features, training, test, classifier) :
    '''
    graph_data:     (str) path to graph data
    abs_data:       (str) path to abstract data
    paper_data:     (str) path to author-paper data
    ls_features:    (list) list of strings that represent the features to use in this pipeline
    training:       (str) path to training.csv
    test:           (str) path to test.csv
    classifier:     (str) Can be one of "LG", "SVM", "RBF", MLP, ... (will add more)
    '''

    # ----------------- ROUGH DRAFT -----------------

    # 1. split training into 5 different sets (for 5-fold cross validation)

    # 2. iterate through each set for CV
    # ------ 2.1 find graph and text features (for training)
    # ------ 2.2 train model on training features
    # ------ 2.3 predict h-index for validation set, store [MSE + model]

    # 3. Find graph and text features for test

    # 4. predict h-index for test, and store predicts as csv under predictions. 
    # ------ Saving conventions: [DATE]-[CLASSIFIER]-[PARAMS]-submission.csv

    # 5. save model (using pickle)

if __name__ == "__main__" :



    X = pd.read_parquet('data/train-graph-text-bert-prone-061221.parquet')
    y = X['hindex']
    X.drop('hindex', axis=1, inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    # train a regression model and make predictions
    norm = Normalizer().fit(X_train)
    X_train_norm = norm.transform(X_train)
    X_val_norm = norm.transform(X_val)

    scaler =  RobustScaler().fit(X_train_norm)
    X_train_scaled = scaler.transform(X_train_norm)
    X_val_scaled = scaler.transform(X_val_norm)

    # reg = Lasso(alpha=0.1)
    # reg = tree.DecisionTreeRegressor()
    # reg = RandomForestRegressor(max_depth=2, random_state=0)
    reg = MLPRegressor(random_state=1, max_iter=500, activation='logistic')
    # reg = Ridge(alpha=0.1)
    # reg = SVR(C=1.0, epsilon=0.2)
    reg.fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_val_scaled)
    
    for i in range(len(y_pred)):
        if y_pred[i] < 0 :
            y_pred[i] = 0

    # write the predictions to file
    print(mean_squared_error(y_val, y_pred))