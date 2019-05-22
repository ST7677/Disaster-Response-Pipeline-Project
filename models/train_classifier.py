# import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - Path used for importing the database     
    OUTPUT
        Returns the following variables:
        X - Input features. Returns the messages column from the dataset
        Y - Categories of the dataset.  This will be used for classification of the input X
        y.keys - Columns of the Y
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df =  pd.read_sql_table('CleanedDataTable', engine)
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y, y.keys()

def tokenize(text):
    '''
    INPUT 
        text: Text that is to be processed
    OUTPUT
        Returns processed text variable (tokenized,  stripped, lower cased, and lemmatized)
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features
        y_train: Training labels
    OUTPUT
        Returns a pipeline model that is tokenized, count vectorized, TFIDTransformed and built into an ML model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, Y_test, category_names):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        Precision, recall and f1-score
    '''
    # Lets predict on test data
    y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))

def save_model(model, model_filepath):
    '''
    This function Saves the model to the disk
    INPUT 
        model: The model to be Saved
        model_filepath: Path for saving the model
    OUTPUT
        Pickle file for the model.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
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
