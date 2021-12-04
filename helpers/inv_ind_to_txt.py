# ----------------------------------------------------------------------------- #
#   Script that will convert all the abstracts from inverted index to strings.  #
# ----------------------------------------------------------------------------- #

import pandas as pd
from datetime import datetime

def abs_to_str(abstract_file, outfile) :
    '''
    abstract_file:      (str) path to abstract.txt
    '''

    start = datetime.now()

    # Opening original abstract file
    abstract = open(abstract_file, 'r')
    abstract = abstract.readlines()
    abstract = pd.DataFrame(abstract, columns=['InvInd'])
    abstract['PID'] = [e.split('----', maxsplit = 1)[0] for e in abstract['InvInd']]
    abstract['InvInd'] = [e.split('----', maxsplit = 1)[1] for e in abstract['InvInd']]

    new_abstracts = abstract.copy()

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
    
    new_abstracts.drop(columns = ['InvInd'], inplace = True)
    new_abstracts.to_parquet(outfile)

    print(f'Total time: {start-datetime.now()}')
