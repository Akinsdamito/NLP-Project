# import libraries
import nltk
import re
import sys
import sqlite3
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import gc 
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score, accuracy_score,hamming_loss
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import ClassifierChain
from nltk.corpus import wordnet
import time
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def load_data(database_filepath):
    
    """
        This function takes in the database path and it reads the data
        
        
        return: The messages which are the predators, the response i.e the labels as a dataframe 
            and an array of all the labels
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM Message_Cat' , con = conn)
    X_data_set=df['message'].tolist()
    Y_data_set = df.drop(['id','message','original','genre'], axis=1)
    label_name=Y_data_set.columns
    return X_data_set,Y_data_set,label_name

def word_count(text):
  
    """
        This function takes in text as a string, and separate it into different words
        
        
        return: each words with their counts 
    """
    c_vec = CountVectorizer(tokenizer=None)
    ngrams = c_vec.fit_transform(text)

    count_values = ngrams.toarray().sum(axis=0)
 
    vocab = c_vec.vocabulary_
 
    df_unigram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Unigram'})
    return df_unigram


def clean_text_process(text,stopwords):
    
    """
        This function takes in text as a string,array of additional words in the text if we want to remove
        them. It replace symbols with space and also remove the stopwords if Location is None, but if
        is not None, at add the array to the Stopwords be perfoming the removal of words.
        
        return: modified text
    """
    
    replace_symbol = re.compile('[/(){}\[\]\|@,;?:\-\.]')
    final_text=[]    
    for  i in text:  

    # lowercase text    
        text = i.lower()
    # Single character removal
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
        text = re.sub(r'\s+', ' ', text)  
      
    # replace replace_symbol symbols by space in text.
        text = replace_symbol.sub(' ',text) 

    # remove symbols which are not in [a-zA-Z_0-9] from text
        text = re.sub(r'\W+', ' ', text)
    
    # remove symbols which are not in numeric from text
        text = re.sub(r'\d', ' ', text)
              
    # remove numbers from text
        text = re.sub('[0-9]', ' ', text)
    #STOPWORDS = stopwords.words('english')
        
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
            
        final_text.append(text)
    return final_text

def preprocess(data_set,STOPWORDS):
    
    """
        This function takes in text and stopwords, it uses the function WORDCOUNT to count the 
        occurence of each word and add word with only one occurence to the stopwords, then uses function 
        CLEAN_DATA to remove the combined stopwords and also preprocesed the data
        
        return: Text devoid of noise.
    """
    # Count of each tokens in the dataset
    start = time.time()
    print("getting less frequent words in dataset ......")
    wordcount=word_count(data_set)
    new_stopword=wordcount[wordcount['frequency']==0]['Unigram'].values.tolist()
    print('collection of words completed.: {} mins'.format(round((time.time()-start)/60 , 2)))
    ## Adding our own stopwords
    STOPWORDS.extend(new_stopword)

    ## De-noising the dataset and normalisation
    print("starting data preprocessing ......")
    clean_data=clean_text_process(data_set,stopwords=STOPWORDS)
    print('data preprocessing completed.: {} mins'.format(round((time.time()-start)/60 , 2)))

    return clean_data

def tokenize(text):
    
    """
        This function seperated the text to different tokens
        
        return: Tokens.
    """
    
    tokens = word_tokenize(''.join(c for c in text ))
    
    return tokens

def build_model():
    
    """
        This function build the model pipeline by using grid search to get
        the best parameters.
        
        return: Pipeline.
    """
    
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),              
                ('mnb', BinaryRelevance(MultinomialNB()))
    ])

    # specify parameters for grid search
    parameters = {
        
        'mnb__classifier': [MultinomialNB()],
        'mnb__classifier__alpha': np.linspace(0.5, 1.5, 2),
               
        
    }

    # create grid search object
    cv =  GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy')
    
    return cv

def evaluate_model(model, X_test, 
                   Y_test, category_names):
    """
        This function evaluate the model 
        
        return: The classification reports for each label.
    """
    
    prediction = model.predict(X_test)
    prediction_df=pd.DataFrame.sparse.from_spmatrix(prediction)
    prediction_df.columns=category_names
    for i in range(prediction_df.shape[1]):
        
        report=metrics.classification_report(Y_test.iloc[:,i], prediction_df.iloc[:,i])        
        print('\033[1m'+'Classification Report for'+'\033[0m',category_names[i].upper())        
        print(report)
        print('\n')
        
def save_model(model, model_filepath):
    
    """
        This function save the model as a pickle file
        
        return: The classification reports for each label.
    """
    filename =  model_filepath
    pickle.dump(model, open(filename, 'wb')) 
STOPWORDS = stopwords.words('english')   
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #STOPWORDS = stopwords.words('english')
        Clean_X=preprocess(X,STOPWORDS)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        try:
            print('Training model...')
            start = time.time()
            model.fit(X_train, Y_train)
            print('Training model.: {} mins'.format(round((time.time()-start)/60 , 2)))
            
        except RuntimeWarning:
            pass
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
