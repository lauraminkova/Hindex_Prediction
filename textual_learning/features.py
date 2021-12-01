# ----------------------------------------------------------------------------- #
#               Script with the different features for text data                #
# ----------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import os
# for text_clustering1
from text_clustering.vectorizer import cluster_paragraphs
from random import shuffle

def number_papers(ls_authors, authors_paper) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    '''

    ls_num_papers = [0] * ls_authors.shape[0]
    
    for a in ls_authors.index :
        author = str(ls_authors.loc[a])
        ls_num_papers = 5 - authors_paper.loc[author].isna().sum().sum()
    
    return ls_num_papers

def total_sum_abs_words(ls_authors, authors_paper, abstracts) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    abstracts:      (DataFrame) dataframe with paperIDs and their inverted index (original file)
    '''
    
    ls_length = [0] * ls_authors.shape[0]

    for a in ls_authors.index :
        author = str(ls_authors.loc[a])
        for pid in authors_paper.loc[author] :
            # Checking paperID exists, and that it's abstract is available to us
            if (pid != None) and (len(abstracts.loc[abstracts.PID == pid, 'InvInd'].values) != 0):
                inv_ind = eval(abstracts.loc[abstracts.PID == pid, 'InvInd'].values[0])
                ls_length[a] += inv_ind['IndexLength'] 
    
    return ls_length


# --------------------- NOVEMBER 30th 11:39 pm update: NOT WORKING ---------------------

def text_clustering1(abstracts, num_clusters) :
    '''
    abstracts:      (DataFrame) dataframe with paperIDs and their inverted index
    num_clusters:   (int)  Number of clusters interested in
    '''

    ls_clusters = [0] * abstracts.shape[0]
    
    # list of all the string abstracts
    ls_strs = list(abstracts.InvertedIndex)
    clusters = cluster_paragraphs(ls_strs, num_clusters)
    print(clusters)

    # for i in range(len(clusters)) :
    #     clust_i = clusters[i]
    #     print(f'----------- CLUSTER {i} -----------')
    #     for j in range(len(clust_i)) :
    #         print(j)
    #     " ------------------------------------------- "



TEXT_FEATURES = {'numpapers': number_papers, 'tot_abs_len': total_sum_abs_words, 'txt_clust1' : text_clustering1}


if __name__ == '__main__' :