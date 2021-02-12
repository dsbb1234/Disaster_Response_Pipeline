import sys

sys.path.append("/home/workspace/models")

# from starting_verb_extractor import StartingVerbExtractor

import json
import plotly
import pandas as pd
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


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table("final_table", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Creating the x and y variables for Graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Creating new dataframe with only the 36 categories
    df2 = df.iloc[:,4:]
    
    # Creating the x and y variables for Graph 2
    category_names = list(df2.columns)
    category_counts = [df2.sum()[i] for i in range(len(df2.columns))]

    
    # Creating the x and y variables for Graph 3
    pcts = [(df.direct_report.value_counts(normalize=True) * 100)[i] for i in range(2)]
    pct_labels = ['non-direct report','direct report']
    
    # create visuals
    graphs = [
        
            # This graph depicts the distribution of genre names by count
        
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='blue')
                )
            ],

            'layout': {
                'title': 'Message Genre Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        
            # This depicts all categories by count 
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(color='blue')
                )
            ],

            'layout': {
                'title': ' Message Category distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'categoryorder':'total ascending'
                }
            }
        },
        
            # This graph compares the percentage of Directly/Indirectly reported messages
        
        {
            'data': [
                Bar(
                    x=pct_labels,
                    y=pcts, 
                    marker=dict(color='blue')
                )
            ],

            'layout': {
                'title': 'Percent of Direct/Indirectly Reported Messages',
                'yaxis': {
                    'title': "Percentages"
                },
            }
        },
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()