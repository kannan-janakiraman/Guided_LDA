# Guided_LDA
Guided LDA Project Repository

Guided LDA is extension to LDA where the top level classes are partitioned in to multiple classes using guided words for each class. 

In this project, the class GMC_LDA in the gmc_lda.py can be used to perform the following functionalities 
- initialize with dataframe and col that contains the corpus 
- generate_class_dataframes - method to partition dataframe in to class level dataframes 
- get_best_model - performs grid search and returns C_v scores for all classes 
- get_best_model_for_class - performs grid search and returns C_v scores for given class idx 
- get_best_LDA_models - builds LDA models for given hyper parameters 



