import sys

sys.path.append("/home/workspace/models")

from starting_verb_extractor import StartingVerbExtractor

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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
                    marker=dict(color='rgb(85,201,159)')
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
                    marker=dict(color="darkorange")
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
                    y=pcts
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