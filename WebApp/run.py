import json
import plotly
import pandas as pd
import sqlite3

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as go
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

conn = sqlite3.connect('../data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM Message_Cat' , con = conn)

# load model
model = pickle.load(open('../models/classifier.pkl', 'rb'))

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

  # first chart plots arable land from 1990 to 2015 in top 10 economies 
  # as a line chart
    graph_one = []
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph_one.append(
      Bar(
                x=genre_names,
                y=genre_counts
      )
    )

    layout_one = dict(title = 'Distribution of Message Genres',
                yaxis = dict(title =  "Count"),
                xaxis = dict(title = "Genre"),
                )
    

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []
    categories = list(df.columns.values)
    
    graph_two.append(
      Bar(
      x = categories[4:],
      y = df.iloc[:,4:].sum().values,
      )
    )

    layout_two = dict(title = 'Number of comments in each category',
                xaxis = dict(title = 'Comment Type ',),
                yaxis = dict(title = 'Number of comments'),
                uniformtext_minsize=8, uniformtext_mode='hide',
                texttemplate='%{text:.2s}', textposition='outside',
                 textangle=75     
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    rowSums = df.iloc[:,4:].sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    multiLabel_counts = multiLabel_counts.iloc[1:]
    graph_three.append(
          Bar(
      x=multiLabel_counts.index,
       y=multiLabel_counts.values,
      )
    )

    layout_three = dict(title = 'Number of comments having multiple labels ',
                xaxis = dict(title = 'Number of labels',),
                yaxis = dict(title = 'Number of comments'),
                )
    
   
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
   

    return figures


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    graphs = return_figures()

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
    classification_labels_result = model.predict([query])[0]
    
    #Convert sparse matrix to array
    classification_labels=pd.DataFrame.sparse.from_spmatrix(classification_labels_result).values
    classification_results = dict(zip(df.columns[4:], classification_labels.flatten()))

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