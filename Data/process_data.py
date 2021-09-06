import re
import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

def load_data(messages_filepath, categories_filepath):
    
    """
        Description -This function takes in the files path and read it into dataframe
        
        
        return: It returns a 2 dataframe. 
   """
    
    messages = pd.read_csv(messages_filepath, sep=",", header=0)
    categories = pd.read_csv(categories_filepath, sep=",", header=0)
    return messages,categories

def clean_data(data_df1,data_df2):
    
    """
        Description -This function takes in two dataframes. The first is the dataframe that 
        has the messages and data_df2 is the categories dataframe. It transformed and merge the 2 
        dataframes.It also clean the dataframe 
        
        
        return: It returns a clean dataframe. 
   """
    
    # merge datasets
    df = pd.merge(data_df1,data_df2,how='inner', on='id')
    
    # create a dataframe of the 36 individual category columns
    data_df2 = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = data_df2.iloc[0]

    #  to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x : x[:-2])) 


    # rename the columns of `categories`
    data_df2.columns = category_colnames


    for column in data_df2:
    # set each value to be the last character of the string
        data_df2[column] = data_df2[column].str[-1]
    
    # convert column from string to numeric
        data_df2[column] = pd.to_numeric(data_df2[column])


    # drop the original categories column from `df`

    df.drop(columns=['categories'],axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,data_df2],axis=1)
    
    new_df=df[(df['related']<2)]
    
#    new_df.drop(labels=['index'], axis=1,inplace=True)

    Unique_df=new_df.drop_duplicates().reset_index(drop=True)    
    return Unique_df


def sql_fetch(conn):

    cursorObj = conn.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    #print(cursorObj.fetchall())
    
def save_data(df, database_filename):
    
    """
        Description -This function save the data into a database
        
        
         
   """
    
    conn = sqlite3.connect(database_filename)
    sql_fetch(conn)

    engine = create_engine('sqlite:///'+database_filename)

    df.to_sql('Message_Cat', con=engine, if_exists='replace', index=False)
     
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        data_set = clean_data(df[0],df[1])
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(data_set, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()    
