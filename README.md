# INF554-Final-Project

We used Python 3.9.7.

Rough steps so far:

1. *Prepping data*
    * function `get_auth_pid_tbl` from  helpers/df_helpers.py: converted author_papers.txt into a human-friendly dataframe (doesn't take too long)
    * function `get_abstract_tbl` from  helpers/df_helpers.py: converted abstracts.txt into a human-friendly dataframe (doesn't take too long)
    * function `abs_to_str` from helpers/inv_ind_to_txt.py: converted inverted indices to strings **(this can take several hours...)**
2. *Prepping features* 

    All features were pre-computed and saved locally to avoid constant re-computing. For the features that required splitting up of the data for faster computation, the not-so-pretty scripts of piecing them back together can be found in the exploration folder. 
    
    * Graph Features (all scripts can be found at features/graph_features.py)
        * Degree
        * Core-number
        * Average neighbour degree
        * Eigen centrality
        * Degree centrality
        * Clustering
        * Triangle
        * **Betweenness centrality** : IF IT EVER FINISHES RUNNING 
        * **Closeness centrality** : IF IT EVER FINISHES RUNNING
    * Text Features (all scripts can be found at features/text_features.py)
        * function `number_papers`: calculates the number of major papers an author has published (does not take too long)
        * function `total_sum_abs_words`: calculates the total sum of words in an author's major papers' abstracts (takes a while... couple of hours)
        * function `get_scibert_vectors`: concatenate the string versions of every abstract an author has, and feed that into the pre-trained scibert model to get vectors of length 798 for every author **(can take a very very long time... recommend to split up authors in several groups)**. Downloaded SciBERT from: https://github.com/allenai/scibert, although you don't really need to if you use HuggingFace Pytorch models.

3. *Pipeline*

    Full pipeline, from prepping data and features to fitting a model can be found in pipeline.py.

    * Scaling / Normalizing
        * For the moment we're currently using scikit-learn's robust scaler
    * Classifiers (can be found under classifiers/classify.py)
    