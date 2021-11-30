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

    ls_num_papers = [] * ls_authors.shape[0]
    
    for a in ls_authors.index :
        ls_num_papers = 5 - authors_paper.loc[[a]].isna().sum().sum()
    
    return ls_num_papers

def total_sum_abs_words(ls_authors, authors_paper, abstracts) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    abstracts:      (DataFrame) dataframe with paperIDs and their inverted index
    '''

    ls_length = [] * ls_authors.shape[0]

    for a in ls_authors.index :
        
        for p in ~ authors_paper.loc[[a]].isna()
        ls_length[]


TEXT_FEATURES = {'numpapers': number_papers, 'tot_abs_len': total_sum_abs_words}
