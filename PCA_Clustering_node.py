import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler,  PowerTransformer, Normalizer
from sklearn.decomposition import PCA
import seaborn as sns
from google.colab import drive
from google.colab import files
drive.mount('/content/drive')

data = pd.read_csv('data/node2vec.csv',sep=' ', header=None) ##to update
data = pd.DataFrame(data)
data = data.to_numpy()


nodes_list = data[:,0]
embeding = data[:,1:]

train_scaled = embeding

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf = principalDf.to_numpy()

kmeans = KMeans(n_clusters=3, random_state=0).fit(principalDf)
labels = kmeans.labels_

nodes_list  = nodes_list.reshape(-1,1)
principalDf = principalDf.reshape(-1,2)
labels = labels.reshape(-1,1)
X_PCA = np.concatenate((nodes_list ,principalDf),axis=1)
clustering = np.concatenate((nodes_list ,labels),axis=1)
X_PCA1 = X_PCA[:,0:2]
X_PCA2 = X_PCA[:,0:3:2]
X_PCA1 = dict(X_PCA1)
X_PCA2 = dict(X_PCA2)
clustering = dict(clustering)
