# ----------------------------------------------------------------------------- #
#               Script with the different features for graph data               #
# ----------------------------------------------------------------------------- #
import networkx as nx
import pickle

# *********************************** #
#            RUN ONLY ONCE            #   
# *********************************** #
def all_networkx_feats(edgelist) :
    '''
    edgelist:   (edgelist file) Path to coauthorship.edgelist
    '''

    G = nx.read_edgelist(edgelist, delimiter=' ', nodetype=int)

    # Basic
    core_number = nx.core_number(G)
    connect = nx.average_neighbor_degree(G)

    #centrality 
    eigen_centrality = nx.eigenvector_centrality(G)
    degree_centrality = nx.degree_centrality(G)

    # clustering 
    clustering = nx.clustering(G)
    triangle = nx.triangles(G)

    # page rank
    pagerank = nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)

    with open("core_number.pkl", "wb") as f :
        pickle.dump(core_number, f)

    with open("connect.pkl", "wb") as f :
        pickle.dump(connect, f)

    with open("eigen_centrality.pkl", "wb") as f :
        pickle.dump(eigen_centrality, f)

    with open("degree_centrality.pkl", "wb") as f :
        pickle.dump(degree_centrality, f)

    with open("clustering.pkl", "wb") as f :
        pickle.dump(clustering, f)

    with open("triangle.pkl", "wb") as f :
        pickle.dump(triangle, f)

    with open("pagerank.pkl", "wb") as f :
        pickle.dump(pagerank, f)