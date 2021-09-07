# NLP-Project
This project perform an ETL pipeline, NLP pipeline and Machine learning pipeline on a dataset .

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#Uploded Files">Uploded Files</a></li>
    
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

# About-the-project
This project is divided into 3 parts,:
* The first part performed an ETL process on two data datasets i.e we mapped the message dataset with the category data set using the id variable of each dataset as the primary key used in joining them. The category variables are sepparated into different columns, which means we have 36 columns all together. After doing this, it is observed that some messages has more than one category(label), which implies that the problem at hand is a multilabel classification problem. After extracting, transforming and joining this data, it is loaded in an sqlite database.
* The second part of the project is NLP pipeline and Machine learning pipeline process. Since it is impossible to train a model with a test data, so we converted the messages to a word of vector before training the machine learning model. Different approaches were used to convert the messages to vectors, but before I dabble into that, some preprocessing were done on the texts, eg. conversion of all the characters to lower case, removing stopwords, removing tags etc. Now to convert to vector, we first used CountVectorizer, TfidfTransformer but the demerit of this approach is the size of the vector. An approach to solve this is the use of wordemmbedding, so we use a pretrained wordemmbedding model Glove and also a Word2vec model trained on our dataset using a vector of dimension 200. But we only present the result for CountVectorizer, TfidfTransformer in this document because of the limitation of environment provided by udacity. Ater the conversion, we train different machine learning models, but we only present the one trained using multilabel agorithm BInaryRelevance.
* Finally, a web application is designed using flask, which can be use to determine the category which a message falls under when the message is inputted on a search field on the webapp. On the webapp, we also have the distribution plot of each of the categories in the dataset and also how the masseges are distributed base on the number of corresponding categories.
*   
# Getting Started
This project codes are written inform of a script which can be run on any idle. It is contained in 3 folders:
* Data: It contains the ETL code and the two data files. The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path. To run ETL pipeline that cleans data and stores in database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* Models: It contains the ML and NLP pipeline codes in the form of a script. There are 2 script, one is name train_classifier.py and the other is train_classifier_with_Lema. The difference is that the script named train_classifier_with_Lema has Lemmetizer in the token function why the other did not have. The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path. To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
* WebApp: It contains all the required codes to run the app.
        

## Pre-requisites
In order to run the code successfully, the following liberaries have to be installed on your notebook
* Pandas, latest version
* Scikit-Multilearn
* Plotly
* Flask
* Seaborn
* mathplotlib
* Numpy
* Sklearn

# Uploded Files
The following file are uploded in this repository:
* The code for the analysis, which is written in python.
* The data files used for the analysis.
* A readme file that explains the information in the data.


# Acknowledgements
* This project wouldn't be possible if not for the Udacity team, who provided the data used and a workstation to run the code.
