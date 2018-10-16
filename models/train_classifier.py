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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import os
import sys


class MessageLengthTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates the length of a single message
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(x) for x in X]).reshape(-1, 1)


class SpecialCharacterCounter(BaseEstimator, TransformerMixin):
    """
    Counts any special Character in a message
    Idea: maybe desperate or urgent messages contain more of those (e.g. exclamation marks etc.)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(re.sub('[ a-zA-Z0-9]', '', x)) for x in X]).reshape(-1, 1)


def load_data(database_filepath):
    '''

    :param database_filepath:
    :return:
    - X:
    - y:
    - category_names:
    '''

    # sqlite:/// is necessary, followed by path where the DB exists.
    # in this case, go back one step and then drill down to the file
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql_table('Messages', con=engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns.values
    y = y.values
    return X, y, category_names


def tokenize(text, lang="english"):
    '''

    :param text:
    :param lang:
    :return:
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

    :param X_train:
    :param y_train:
    :return:
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
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=50, learning_rate=0.5)))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def display_results(category_names, y_test, y_pred):
    '''

    :param category_names:
    :param y_test:
    :param y_pred:
    :return:
    '''
    for i, cat in enumerate(category_names):
        metrics =  classification_report(y_test[i], y_pred[i])
        print("""Category: {}
              {}
              ----------------------------------------------------
              """.format(cat, metrics))


def save_model(model, folder='classifiers/', filename=''):
    '''

    :param model:
    :param folder:
    :param filename:
    :return:
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

    :return:
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
        display_results(category_names, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_folder + 'AdaBoostClassifier.pkl'))
        save_model(model, model_folder)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()