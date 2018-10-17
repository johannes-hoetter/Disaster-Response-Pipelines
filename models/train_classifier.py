import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


import pickle
import os
import sys

from models.custom_estimators import MessageLengthTransformer, SpecialCharacterCounter


def load_data(database_filepath, table_name='Messages'):
    '''
    Load the Data for Machine Learning from a sqlite Database.
    :param database_filepath: Path to the sqlite Database
    :param table_name: name of the table containing the data
    :return:
    - X: Data for the ML Model (Input Variables)
    - y: Data for the ML Model (Labels / Output Variables)
    - category_names: Names of the Categories which the Table contains
    '''

    # sqlite:/// is necessary, followed by path where the DB exists.
    # in this case, go back one step and then drill down to the file
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql_table(table_name, con=engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns.values
    y = y.values
    return X, y, category_names


def tokenize(text, lang="english"):
    '''
    Prepares a String to be processed by a CountVectorizer for ML-Algorithms.
    Uses: Normalization, Tokenization, Stop Words Removal, Stemming and Lemmatization.
    :param text: input text which isn't processed yet
    :param lang: language of the text file (needed for Stop Words Removal)
    :return: processed text as a list
    '''

    # Normalize (Transform to lowercase and remove any punctuation)
    text = text.lower()
    pattern = re.compile('[^a-z0-9]+')
    text = pattern.sub(' ', text)

    # Tokenize (put text to list)
    words = word_tokenize(text)

    # Remove Stop Words (e.g. 'This', 'the', ...)
    words = [word for word in words if word not in stopwords.words(lang)]

    # Stem / Lemmatize (e.g. 'tokenizing' -> 'token')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='v') for word in words]

    return words


def build_model(X_train, y_train):
    '''
    Builds a ML Pipeline containing a classification estimator.
    The specific parameters for the given estimator (RandomForestClassifier) have
    been calculated by a GridSearchCV during development (see seperate file "ML Pipeline Preparation.ipynb".
    The Pipeline gets fit on the given Training Data.
    :param X_train: Training Data for the ML Model (Input Variables)
    :param y_train: Training Data for the ML Model (Labels / Output Variables)
    :return: fitted pipeline
    '''

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('msg_length', MessageLengthTransformer()),
            ('special_char', SpecialCharacterCounter())
        ])),
        ('scaler', StandardScaler(with_mean=False)), #ML Algorithms often work better with Standardization
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, criterion='gini')))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, category_names, X_test, y_test):
    '''
    Calculates Precision, Recall, F1-Score and Support of the Prediction Results for each possible
    Category.
    :param model: Model used for prediction
    :param category_names: contains the different possible Categories
    :param X_test: Testing Data for the ML Model (Input Variables)
    :param y_test: Testing Data for the ML Model (Labels / Output Variables)
    :return: -
    '''

    y_pred = model.predict(X_test)
    for i, cat in enumerate(category_names):
        metrics =  classification_report(y_test[i], y_pred[i])
        print("""Category: {}
              {}
              ----------------------------------------------------
              """.format(cat, metrics))


def save_model(model, folder='classifiers/', filename=''):
    '''
    Saves the model as a pickle file.
    :param model: Model to be saved
    :param folder: Directory where the model gets saved
    :param filename: Filename of the new pickle file
    :return: -
    '''

    # Input Handling
    if filename == '':
        filename = model.named_steps['clf'].estimator.__class__.__name__  # Name of the used Classification Algorithm
    if folder.endswith('/'):
        path = folder + filename
    else:
        path = folder + '/' + filename

    # Check if given folderpath exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the Model in the given folder
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def main():
    '''
    Runs the script if the file is called directly.
    :return: -
    '''
    if len(sys.argv) == 3:
        database_filepath, model_folder = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, category_names, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_folder + 'RandomForestClassifier.pkl'))
        save_model(model, model_folder)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()