import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

# Create a graph
graph = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)

# Precompute probabilities and generate walks - 
node2vec = Node2Vec(graph, dimensions=5, walk_length=3, num_walks=50, workers=1)  

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings
model.wv.save_word2vec_format('node2vec.csv')

# Save model for later use
model.save('model')
