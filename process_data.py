import re
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



def transform_df(data_df1,data_df2):
    
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

    Unique_df=new_df.drop_duplicates().reset_index(drop=True)    
    return Unique_df


def sql_fetch(conn):

    cursorObj = conn.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    #print(cursorObj.fetchall())
    
if __name__ == '__main__':
        
    # load messages dataset
    print('Loading message file....')
    messages = pd.read_csv('../messages.csv', sep=",", header=0)

    # load categories dataset
    print('Loading categories file....')
    categories = pd.read_csv('../categories.csv', sep=",", header=0)
     
    data_set=transform_df(messages,categories) 
    print('Cleaning and data tranformation done.')
    
    print('Saving data on sqlite database....')
    conn = sqlite3.connect('DisasterResponse.db')

    sql_fetch(conn)

    engine = create_engine('sqlite:///DisasterResponse.db')

    data_set.to_sql('Message_Cat', con=engine, if_exists='replace', index=False)
    print('Saving completed')