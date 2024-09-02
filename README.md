# Guided_LDA
Guided LDA Project Repository

Guided LDA is extension to LDA where the top level classes are partitioned in to multiple classes using guided words for each class. 

In this project, the class GMC_LDA in the gmc_lda.py can be used to perform the following functionalities 
- initialize with dataframe and col that contains the corpus 
- generate_class_dataframes - method to partition dataframe in to class level dataframes 
- get_best_model - performs grid search and returns C_v scores for all classes 
- get_best_model_for_class - performs grid search and returns C_v scores for given class idx 
- get_best_LDA_models - builds LDA models for given hyper parameters 

You can refer to GuidedLDA.ipynb for example usage. The tweets used for the example is stored in filtered_tweets.csv


Sample usage of the class is given below for tweets sent to AppleSupport. 

```
import gmc_lda

# create gmc_model object with dataframe 
gmc_model = GMC_LDA(df = tweets_filtered_df, raw_col = 'text_x')

# partition the dataframe to multiple classes 
noise_words = ['hello', 'hey', 'apple','hi', 'fuck', 'shit', 'help','issu', 'problem', 'work','batteri','time', 'work' ,'thanks', 'letter', 'glitch', 'question', 'mark', 'turn', 'need','help','problem','issu', 'shit'] 
class_common_words = ['appl', 'battery', 'update', 'music']
class_words = [ ['iphone', 'phone', 'iphonex','youtube', 'mail','facebook','twitter'],
                ['ipad', 'tablet'],
                ['mac', 'macbook', 'sierra', 'macos', 'osx', 'keyboard'], 
                ['ipod', 'itunes', 'music','podcast'],
                ['watch', 'watchos', 'series'],
                ['icloud', 'drive', 'pages', 'numbers', 'keychain', 'photos', 'apple id'], 
                []]
class_names = ['iphone', 'ipad', 'mac','ipod', 'watch', 'icloud', 'others']

gmc_model.generate_class_dataframes(noise_words=noise_words,
                                    class_words=class_words,
                                    class_names=class_names)
gmc_model.print_partition_info()

# perform class level LDA 
gmc_model.get_best_model(topic_only=False)
```
