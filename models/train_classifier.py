# import relevant libraries to support this machine learning pipeline model

import sys
# import the numpy and pandas libraries
import pandas as pd
import numpy as np

# import sqlalchemy to support importing the SQL database
from sqlalchemy import create_engine

# import these libraries to do the pre-cleaning tokenizing and lemmatizing of the text files
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import these files to support the pipeline and machine learning models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

# import pickle to save the classifier model as a pickle file
import pickle


def load_data(database_filepath):
    '''
    This function reads the SQL database
    
    Inputs:
    
    Arguments:
    database_filepath - identifes the file path and file name to be loaded
    
    
    Outputs:
    
    df - a dataframe containing the SQL data from the ETL pipeline
    
    '''
    # load data from SQL database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("final_table", engine)
    
    # This code remaps the values of "2" to "1". 
    df['related'] = df.related.map(lambda x: 1 if x==2 else x)

    # create a copy to work on
    df_copy = df.copy(deep=False)
   
    # since the identification of the message is considered the target, the message must be the feature.  
    # assign the message to x and assign all remaining rows beyond the offer variable to the y dataframe.
    X = df_copy.message
    y = df_copy.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''This class takes a string of tokens as its input and classifies each word as a noun, verb, etc.
    
    Inputs:
    BaseEstimator
    
    TransformerMixin
    
    
    
    Outputs:
    X_tagged - a dataframe containing the tagged words
    
    
    '''

    def starting_verb(self, text):
        '''
        This function takes words and classifies each word as either verb or not.
    
        Inputs:
        text = the string containing the words
    
    
        Outputs:
        A 1 if the word is a verb or a 0 it it is not a verb 
        
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        '''
        This fits the function to x or x and y
    
        Inputs:
        x, y - the series to apply the function to
     
    
    
        Outputs:
        A 1 if the word is a verb or a 0 it it is not a verb 
        
        '''
        return self

    def transform(self, X):
        '''
        This applies the functions defined above to the series and returns the data as a dataframe
    
        Inputs:
        X
     
    
    
        Outputs:
        X_tagged - returns the tagged dataframe
        
        '''      
        
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
    This function breaks a stream of text into words, phrases and symbols, then removes the symbols to leave words.
    
    Inputs:
    text - the stream of text to be tokenized
    
   
    Outputs:
    clean_tokens - lower case words with special symbols and extra spaces removed.
    
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# this code modifies some parameters in the classification model to optimize it


def build_model():
    """
    
    This function takes a list of parameters that we want to load into the model and the pipeline.  It then builds the model and     
    launches GridsearchCV
        - Tokenize text
        
        Args:
            No arguments are passed into the function
            
        Returns:
            cv - the model that will be used to fit the data
    """
    
    parameters = { }
    
    # load the pipeline to be used in this version of the model
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, max_df = 0.5)),
                ('tfidf', TfidfTransformer(use_idf=1))
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])



               
    # this code launches the GridSearchCV code to evaluate the models in the pipeline
    cv  =GridSearchCV(pipeline, param_grid=parameters)

    
    return cv


def display_results(model, y_test, y_pred, cat_nam):
    '''
    This function displays the results of the model, shows the test results in summary by label 
    and displays the F1 test results and best parameters using the confusion_matrix function.
    
    Inputs:
    model - result of GridSearchCV
    
    y_test - the test array
    
    y_pred - the predictions resulting from the GridSearch
    
    Outputs:
    
    Summary of test results by label
    
    F1 test results
    
    Best parameters for the model
    
    '''
    
    print(classification_report(y_test, y_pred, target_names=cat_nam))
    print(model.best_params_)




def save_model(model, model_filepath):
    
    '''This function saves the model
    
    Inputs:
    model - the fitted model
    
    model_filepath - the path to the model file
    
    Outputs:
    
    The saved pickle file

    
    '''
    
    # this code saves the classfier as a pickle file
    pkl_filename = model_filepath 
    with open(pkl_filename, 'wb') as file: 
        pickle.dump(model, file)


def main():
    '''
    This function executes the defined functions above in a specific order:
    1) import the data
    2) tokenize, memmatize the data and use StartingVerbExtractor to tag the words
    3) create the model using GridsearchCV
    4) fit the model
    5) predict labels from test data
    6) evaluate resuls using confusion matrix
    7) save the model as a pickle file
    
    Inputs:
    
    Arguments:
     
    None
    
    Outputs:
    
    A summary of results by category and overall results showing precision, recall, F1-score , number of data points
    
    Best features based on the parameters loaded
    
    The saved pickle file
    
    '''    
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        cat_nam = y.columns
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        display_results(model, y_test, y_pred, cat_nam)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()