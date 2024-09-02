import re
import contractions
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd 
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel 
from tqdm.notebook import tqdm
import numpy as np
import sys

'''
Class to encapsulate the GMC_LDA algorithm and the dataset and data results
'''
class GMC_LDA:

    def __init__(self, df, raw_col) -> None:
        # common properties 
        self.df = df 
        self.raw_col = raw_col
        self.clean_col = 'cleaned_tweets_for_gmc_lda'
        # properties for data pre-processing 
        self.stemmer = PorterStemmer()
        self.lemmer = WordNetLemmatizer()
        nltk.download("words")
        self.stop_words = set(stopwords.words('english'))
        nltk.download('wordnet')
        # properties to split the dataset 
        self.class_df_array = None
        self.class_best_lda_models = None
        self.class_words = None 
        self.class_names = None
        self.noise_words = None
        self.best_class_model_configs = []
        self.class_corpus = []
        self.class_dictionary = []
        self.best_LDA_models = []
        
    def pre_process(self, tweet, filter_words):
        '''
        Method to pre process document line including case change, removal of non-alphabetic 
        characters, smaller words, stop words and then stemming it. 

        Input   : tweet text as String 
        Returns : pre-processed line as String 
        '''
        tweet = tweet.lower() # convert to lower case
        tweet = re.sub("@\w*"," ", tweet) # remove mentions 
        tweet = re.sub('https?://(?:www\.)?[a-zA-Z0-9./]+', '', tweet) # remove URLs
        tweet = re.sub("[!'’\"$%&()*+-/:;<=>?@[\]^_`{|}~\n -' ]", " ", tweet) # remove punctuation 
        tweet = tweet.replace('“', '').replace('”', '').replace('"', '') # remove fancy quotes 
        tweet = re.sub('[0-9]+\w*','', tweet) # remove numbers
        emojis = re.compile("[" # remove emoticons & symbols
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags = re.UNICODE)
        tweet = emojis.sub(r' ', tweet)
        tweet = re.sub('(?<![\w\-])\w(?![\w\-])','',tweet) 
        tokens = tweet.split()
        tokens = [i for i in tokens if not i in self.stop_words and not i in filter_words] # remove stop words 
        cleaned = " ".join(tokens)
        cleaned = contractions.fix(cleaned) # expand colloquial use abbreviations 
        result = []
        for token in gensim.utils.simple_preprocess(cleaned) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                converted = self.stemmer.stem(self.lemmer.lemmatize(token,pos='v'))
                if converted not in filter_words:
                    result.append(converted)
        return " ".join(result)

    def generate_class_dataframes(self, noise_words, class_words, class_names):
        
        self.noise_words = noise_words
        self.class_words = class_words
        self.class_names = class_names

        self.df[self.clean_col] = self.df[self.raw_col].apply(self.pre_process, args=([],))
        self.class_df_array = [] 
        rem_df = self.df 
        for class_words in self.class_words:
            words_match = '|'.join(class_words)
            filtered = rem_df[rem_df[self.raw_col].str.lower().str.contains(words_match)| rem_df[self.clean_col].str.contains(words_match)]
            self.class_df_array.append(filtered.reset_index())
            rem_df = pd.merge(rem_df,filtered, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        # self.class_df_array.append(rem_df)
        self.df.drop(['cleaned_tweets_for_gmc_lda'], axis=1, inplace=True)
        return 
    
    def print_partition_info(self):
        for i, df in enumerate(self.class_df_array):
            print(self.class_names[i], df.shape)
    
    def clean_class_tweets (self, df, noise_words = []):
        df[self.clean_col] = df[self.raw_col].apply(self.pre_process, args=(noise_words,))
        return df 

    def vectorize_class_tweets(self, df):
        tweets = df[self.clean_col]
        tweets_tok = tweets.apply(lambda x: x.split())
        tweets_dictionary = Dictionary(tweets_tok)
        tweets_dictionary.filter_extremes(no_below=5, no_above=0.5)
        tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tweets_tok]
        return tweets_dictionary, tweets_corpus 

    def run_class_LDA_model(self, df, corpus, dictionary, t, a, b):
        lda_model = LdaMulticore(corpus=corpus,
                                id2word=dictionary,
                                num_topics=t,
                                passes=10,
                                alpha=a, 
                                eta=b,
                                chunksize=100,
                                per_word_topics=True, 
                                random_state=42
                                ) 
        list_of_tweets = df[self.clean_col].to_list()
        tweet_texts = [tweet.split() for tweet in list_of_tweets]
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                            texts=tweet_texts,
                                            dictionary=dictionary, coherence='c_v')
        return coherence_model_lda.get_coherence()
    
    def get_lda_model_for_best_config(self, corpus, dictionary ):
        lda_model = LdaMulticore(corpus=corpus,
                                id2word=dictionary,
                                num_topics=t,
                                passes=10,
                                alpha=a, 
                                eta=b,
                                chunksize=100,
                                per_word_topics=True, 
                                random_state=42
                                ) 

    def get_best_class_model(self, df, corpus, dictionary, topic_only):

        # Topics range
        min_topics = 2
        max_topics = 10
        step_size = 1
        topics_range = range(min_topics, max_topics+1, step_size)

        # Alpha parameter
        if topic_only is True:
            alpha = ['symmetric']
        else:
            alpha = list(np.arange(0.01, 1, 0.3))
            alpha.append('symmetric')
            alpha.append('asymmetric')

        # Beta parameter
        if topic_only is True:
            beta = [0.61]
        else:
            beta = list(np.arange(0.01, 1, 0.3))
            beta.append('symmetric')

        # results placeholder
        model_results = {'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'Coherence': []
                        }

        # tqdm progress bar
        pbar = tqdm(total=(len(beta)*len(alpha)*len(topics_range)),
                        file=sys.stdout, colour='green')

        ### takes ~ 37 minutes
        for k in topics_range:    # iterate through validation corpuses
            for a in alpha:       # iterate through alpha values
                for b in beta:    # iterare through beta values
                    # get the coherence score for the given parameters
                    cv = self.run_class_LDA_model(df,corpus,
                                    dictionary=dictionary,
                                    t=k, a=a, b=b)
                    # Save the model results
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    # update tqdm progress bar
                    pbar.update(1)
                    pbar.refresh()
        pbar.close()

        # save the results to a dataframe
        results_df = pd.DataFrame(model_results)
        return results_df
    
    def get_best_model(self, topic_only = True):
        results = []
        for idx, df in enumerate(self.class_df_array):
            cleaned_df = self.clean_class_tweets(df, self.noise_words)
            dictionary, corpus = self.vectorize_class_tweets(cleaned_df)
            class_result = self.get_best_class_model(cleaned_df, corpus, dictionary, topic_only)
            self.best_class_model_configs.append(class_result)
            print(self.class_names[idx])
            print(class_result.sort_values(by='Coherence', ascending=False).head())
            results.append(class_result)
        return results
    
    def get_best_model_for_topic(self, topic_idx, topic_only = True):
    
        cleaned_df = self.clean_class_tweets(self.class_df_array[topic_idx], self.noise_words)
        dictionary, corpus = self.vectorize_class_tweets(cleaned_df)
        class_result = self.get_best_class_model(cleaned_df, corpus, dictionary, topic_only)
        # self.best_class_model_configs.append(class_result)
        print(self.class_names[topic_idx])
        print(class_result.sort_values(by='Coherence', ascending=False).head())
        # results.append(class_result)
        return
    
    def get_best_LDA_models(self, params): 

        for idx, df in enumerate(self.class_df_array):
            cleaned_df = self.clean_class_tweets(self.class_df_array[idx], self.noise_words)
            dictionary, corpus = self.vectorize_class_tweets(cleaned_df)
            best_model = LdaMulticore(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=params[idx][0], 
                                    random_state=42,
                                    chunksize=100,
                                    passes=100,
                                    alpha=params[idx][1],
                                    eta=params[idx][2])
            self.class_corpus.append(corpus)
            self.class_dictionary.append(dictionary)
            self.best_LDA_models.append(best_model)
            x=best_model.show_topics(formatted=False)
            topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
            print(self.class_names[idx])
            for topic,words in topics_words:
                print(" ".join(words))


        
        
