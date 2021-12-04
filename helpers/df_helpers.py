# ----------------------------------------------------------------------------- #
#                  Script with all of the helper functions                      #
# ----------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

def get_auth_pid_tbl(auth_pid_file) :
    '''
    auth_pid:   (txt file)  Path to original "authorID-PID1-PID2-..." file
    '''

    author_papers = open(auth_pid_file, "r")
    author_papers = author_papers.readlines()
    author_papers = pd.DataFrame(author_papers,columns=['Col'])
    author_papers[['author','Papers']] = author_papers['Col'].str.split(':',expand=True)
    author_papers[['P1','P2','P3','P4','P5']] = author_papers['Papers'].str.split('-',expand=True)
    author_papers = author_papers.replace({'\n':''}, regex=True)
    author_papers.index = author_papers['author']
    author_papers.drop(columns = ['Col', 'Papers', 'author'], axis = 1, inplace = True)

    return author_papers

def get_abstract_tbl(abstract_file) :
    '''
    abs:    (txt file) Path to original abstract file 
    '''

    abstract = open(abstract_file, 'r')
    abstract = abstract.readlines()
    abstract = pd.DataFrame(abstract, columns=['InvInd'])
    abstract['PID'] = [e.split('----', maxsplit = 1)[0] for e in abstract['InvInd']]
    abstract['InvInd'] = [e.split('----', maxsplit = 1)[1] for e in abstract['InvInd']]

    return abstract


