import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pylab as plb
import os 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

##Import of the test.csv
df_train = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Machine learning/data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

##Import of the graph
G = nx.read_edgelist('/content/drive/MyDrive/Colab_Notebooks/Machine learning/data/coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 

############# Feature extraction ###############
core_number = nx.core_number(G)

#Basic
core_number = nx.core_number(G)
connect = nx.average_neighbor_degree(G)

#centrality 
eigen_centrality = nx.eigenvector_centrality(G)
degree_centrality = nx.degree_centrality(G)

# clustering 
clustering = nx.clustering(G)
triangle = nx.triangles(G)

#PageRank
pagerank = nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)

############ Features to X_train ################
X_train1 = np.zeros((n_train, 7))
y_train1 = np.zeros(n_train)
X_train1save = X_train1

for i,row in df_train.iterrows():
    node = row['author'] #node = author id
    X_train1[i,0] = G.degree(node) 
    X_train1[i,1] = core_number[node]
    X_train1[i,2] = eigen_centrality[node]
    X_train1[i,3] = degree_centrality[node] 
    X_train1[i,4] = clustering[node] 
    X_train1[i,5] = triangle[node]
    X_train1[i,6] = pagerank[node]
    y_train1[i] = row['hindex']

scaler = PowerTransformer()
# transform data
X_train1 = scaler.fit_transform(X_train1)

condlist = [X_train1[:,3]<first, ((X_train1[:,3]>first) & (X_train1[:,3]< second)), ((X_train1[:,3]>second) & (X_train1[:,3]<third)), X_train1[:,3]>third]
choicelist = [0,1,2,3]
X_train1[:,3] = np.select(condlist,choicelist)
