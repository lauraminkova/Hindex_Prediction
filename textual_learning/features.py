# ----------------------------------------------------------------------------- #
#               Script with the different features for text data                #
# ----------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import os

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


TEXT_FEATURES = {'numpapers': number_papers, 'tot_abs_len': total_sum_abs_words}


if __name__ == '__main__' :