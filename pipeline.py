# ------------------------------------------------------------------- #
#  Script that will run our classification pipeline from end to end   #
# ------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import random
import classifiers.classify as cl
from sklearn.metrics import mean_squared_error

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
