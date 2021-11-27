# ----------------------------------------------------------------------------- #
#   Script that will convert all the abstracts from inverted index to strings.  #
# ----------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import os
from datetime import datetime

def abs_to_str(abstract_file) :
    '''
    abstract_file:      (str) path to abstract.txt
    '''
    start = datetime.now()

    # abstract = pd.read_fwf(abstract_file, header=None, names=["Col"])
    # abstract['Col'] = abstract['Col'].astype('string')
    # abstract[['Paper_id','Inverted_index']] = abstract['Col'].str.split('----',expand=True,n=1)
    # abstract.index = abstract.Paper_id
    # abstract.drop(abstract.columns[[0, 1]], axis = 1, inplace = True)
    abstract = open(abstract_file, 'r')
    abstract = abstract.readlines()
    abstract = pd.DataFrame(abstract, columns=['InvInd'])
    abstract['PID'] = [e.split('----', maxsplit = 1)[0] for e in abstract['InvInd']]
    abstract['InvInd'] = [e.split('----', maxsplit = 1)[1] for e in abstract['InvInd']]

    new_abstracts = abstract.copy()

    print("hi im here")

    for pid in abstract.index :
        # Converting inverted index from str to dictionary
        inv_ind = eval(abstract.loc[pid].InvInd)
        ls_txt = [""] * inv_ind['IndexLength'] 
        for word in inv_ind['InvertedIndex'].keys() :
            ls_ind = list(inv_ind['InvertedIndex'][word])
            for i in ls_ind :
                ls_txt[i] = word
        str_txt = " ".join(ls_txt)
        new_abstracts.loc[pid, 'InvertedIndex'] = str_txt
    
    print(new_abstracts)

    # new_abstracts.to_csv("data/new_abstracts.txt")
    print(f'Total time: {start-datetime.now()}')

if __name__ == '__main__' :

    os.chdir('/home/laura/Documents/Polytechnique/MScT - M1/INF554 Machine Learning/Kaggle Data Challenge/INF554-Final-Project')
    abs_to_str('data/abstracts.txt')
