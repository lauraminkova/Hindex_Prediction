import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

random.seed(123)

def cv(train, y_train, model, params_dict) :
    '''

    '''
    if model == MLPRegressor :
        norm = Normalizer().fit(train)
        train = norm.transform(train)

    scaler =  RobustScaler().fit(train)
    train = scaler.transform(train)
    
    reg = model(**params_dict)

    # reg = Ridge(alpha=0.1)
    # reg = SVR(C=1.0, epsilon=0.2)
    # reg = KNeighborsRegressor(n_neighbors=2)
    scores = cross_val_score(reg, train, y_train, cv=10, scoring='neg_mean_squared_error')
    
    print(f'ALL SCORES: {-1*scores}')
    print(np.mean(scores))
