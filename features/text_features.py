# ----------------------------------------------------------------------------- #
#               Script with the different features for text data                #
# ----------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall


nltk.download("stopwords") 

def number_papers(ls_authors, authors_paper) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    '''

    ls_num_papers = [0] * ls_authors.shape[0]
    
    for a in ls_authors.index :
        author = str(ls_authors.loc[a])
        ls_num_papers[a - min(ls_authors.index)] = 5 - authors_paper.loc[author].isna().sum().sum()
    
    return ls_num_papers

def total_sum_abs_words(ls_authors, authors_paper, abstracts) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    abstracts:      (DataFrame) dataframe with paperIDs and their inverted index (original file)
    '''
    
    ls_length = [0] * ls_authors.shape[0]

    for a in range(ls_authors.shape[0]) :
        author = str(ls_authors.iloc[a])
        for pid in authors_paper.loc[author] :
            # Checking paperID exists, and that it's abstract is available to us
            if (pid != None) and (len(abstracts.loc[abstracts.PID == pid, 'InvInd'].values) != 0):
                inv_ind = eval(abstracts.loc[abstracts.PID == pid, 'InvInd'].values[0])
                ls_length[a - min(ls_authors.index)] += inv_ind['IndexLength'] 
    
    return ls_length

def get_scibert_vectors(ls_authors, authors_paper, abstracts) :
    '''
    ls_authors:     (Serie-like) list of the authorIDs of interest
    authors_paper:  (DataFrame) dataframe with authorIDs and their corresponding paperIDs
    abstracts:      (DataFrame) dataframe with paperIDs and their string abstracts (modified abstracts.txt)
    '''  
    # Loading SciBERT
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    # Loading stop words
    stop_words = set(stopwords.words("english"))

    ls_vectors = []

    for a in range(ls_authors.shape[0]) :

        # This part concatenates all of the abstracts of an author
        author = str(int(ls_authors.iloc[a].values[0]))
        all_abstracts = ""
        for pid in authors_paper.loc[author] :
            if (pid != None) and (len(abstracts[abstracts.PID == pid]['InvertedIndex'].values) != 0):
                abs = abstracts[abstracts.PID == pid]['InvertedIndex'].values[0]
                all_abstracts += abs + " "
        
        # Removing stop words
        filtered_list = [word for word in all_abstracts.split() if word.casefold() not in stop_words]
        all_abs_no_stopwrd = (" ").join(filtered_list)
        
        # This part makes the word embeddings for the abstracts
        inputs = tokenizer(all_abs_no_stopwrd, return_tensors="pt")
        if(inputs['input_ids'].shape[1] > 512) :
            inputs['input_ids'] = inputs['input_ids'][:, :512]
            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
            inputs['attention_mask'] = inputs['attention_mask'][:, :512]
        outputs = model(**inputs)
        tens = outputs.pooler_output.detach().numpy()
        tens = tens.reshape(tens.shape[1])
        ls_vectors.append(tens)
    
    return ls_vectors

def clarans_bert(df_data) :
    # Code adapted from: 
    # https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4 
    '''
    df_data:    (DataFrame) training or test data with bert vectors only 
    '''

    data = df_data.to_numpy()
    data = data.tolist()

    clarans_instance = clarans(data, 100, 6, 4)

    #returns the clusters 
    clusters = clarans_instance.get_clusters()

    #returns the mediods 
    medoids = clarans_instance.get_medoids()
    
    return clusters, medoids


import pickle 

if __name__ == "__main__" :


    # Load your data:
    new_abs = pd.read_parquet('data/new_abstracts.parquet')

    author_papers = open("data/author_papers.txt", "r")
    author_papers = author_papers.readlines()
    author_papers = pd.DataFrame(author_papers,columns=['Col'])
    author_papers[['author','Papers']] = author_papers['Col'].str.split(':',expand=True)
    author_papers[['P1','P2','P3','P4','P5']] = author_papers['Papers'].str.split('-',expand=True)
    author_papers = author_papers.replace({'\n':''}, regex=True)
    author_papers.index = author_papers['author']
    author_papers.drop(columns = ['Col', 'Papers', 'author'], axis = 1, inplace = True)

    # Load test or training data
    training = pd.read_csv('data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
    # Optionally divide test/training data to parallelize the task (for much much quicker results... we split them up into 6)
    X_train_sub = training[145202:]

    # For SciBERT vectors
    ls_vectors = get_scibert_vectors(X_train_sub, author_papers, new_abs)
    with open("bert_training_6.pkl", "wb") as g :
        pickle.dump(ls_vectors, g)