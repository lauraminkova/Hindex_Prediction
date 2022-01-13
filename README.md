# INF554-Final-Project

If you would like to test out our model, feel free to check out the "Running our model" section. If you're more interested in *how* we gathered our features and tuned our model checked out the "Preparing data & model tuning" 

### **Running our model**

In order to test our model, follow the following steps:
1.  Create a conda virtual environment from our "our_env.yml" file. This will install all the necessary packages for the full project. Run `conda env create -f our_env.yml`.
2. Download the prepared training and test dataframes, as well as the submission template from the zip folder we provided in our project submission. WeLink: https://we.tl/t-aAiXrtjWks 
3. Choose which sklearn regressor you'd like to use. We've already imported LogisticRegression, Lasso, Ridge, SVR, MLP, RandomForestRegressor, but feel free to import one of your own!
For our Kaggle results we used MLPRegressor and we use its best parameters which were calculated through our Bayesian hyperparameter tuning script. The parameters were:
`{'activation': 'logistic',
 'alpha': 0.001,
 'early_stopping': True,
 'hidden_layer_sizes': 500,
 'learning_rate': 'invscaling',
 'learning_rate_init': 0.0001,
 'verbose': True}`
 4. Run the command `python3 pipeline.py`.

### **Preparing data & model tuning**


1. *Prepping data*
    * function `get_auth_pid_tbl` from  helpers/df_helpers.py: converted author_papers.txt into a human-friendly dataframe (doesn't take too long)
    * function `get_abstract_tbl` from  helpers/df_helpers.py: converted abstracts.txt into a human-friendly dataframe (doesn't take too long)
    * function `abs_to_str` from helpers/inv_ind_to_txt.py: converted inverted indices to strings **(this can take several hours...)**
2. *Prepping features* 

    All features were pre-computed and saved locally to avoid constant re-computing. For the features that required splitting up of the data for faster computation, the not-so-pretty scripts of piecing them back together can be found in the exploration folder. 
    
    * Graph Features (all scripts can be found at features/graph_features.py)
        * Degree :  it  represents  the  number  ofconnections of each nodes.
        * Core-number : Returns the core numberfor each vertex of rank k which is the maximalnumber  of  subgraph  that  contains  nodes  ofdegree k or more.
        * Eigen centrality : this  measure  emphasizes connection  to  important  nodes  (which  themselves have many connections) over connection to isolated (or less connected) nodes
        * Degree centrality : the degree centrality represents the number of connections of each node with respect to the total number of nodes.
        * Clustering :  represents the fraction of possible  triangles  through  that  node  that  exist with respect to the degree of the node.
        * Triangle :  The  Triangle  score  counts the  number  of  triangles  for  each  node  in  thegraph. A triangle is a set of three nodes whereeach node has a relationship to the other two
        * function `node2vec_emb`: node embedding of the graph to a 5 dimensional space using the Node2Vec framework. 
        * function `prone_emb` : node embedding of the graph to a 32 dimensional space using the ProNe framework. 
        * function `graph_clustering` : Use DBSCAN, a non linear clustering method used to find 2 cluster in the Node2Vec embedding
    * Text Features (all scripts can be found at features/text_features.py)
        * function `number_papers`: calculates the number of major papers an author has published (does not take too long)
        * function `total_sum_abs_words`: calculates the total sum of words in an author's major papers' abstracts (takes a while... couple of hours)
        * function `get_scibert_vectors`: concatenate the string versions of every abstract an author has, and feed that into the pre-trained scibert model to get vectors of length 798 for every author **(can take a very very long time... recommend to split up authors in several groups)**. Downloaded SciBERT from: https://github.com/allenai/scibert, although you don't really need to if you use HuggingFace Pytorch models.
3. *Hyperparameter Tuning*:

    In order to find out what hyperparameters work best for our problem and a given regression model, we ran the `bayesian_hopt.py` script found in the regressors directory. Choose the regressor you're interested in from a list of LogisticRegression, Lasso, Ridge, RandomForestRegressor, KNeighborsRegressor, SVR and MLPRegressor. To run the script you can use the example template given at the bottom of the script. Make sure to have a params and params/trials folder so that you can save your results!

4. *K-Fold Cross-Validation*:

    In order to prevent overfitting and make sure our model was generalizable, we use 10-fold cross validation which can be found in regressors/cross_validation.py

    
