
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pickle
import re
import sys
import warnings
import nltk
import numpy as np
import pandas as pd

from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='disasterResponseTable', con=engine)

    category_name = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_name].values

    return X, y, category_name


def tokenize(text):
    '''
    The code above detect urls then normalize and tokenize the text, removing stop words and apply a lemmatize process.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def build_model():
    '''
    
    '''
    # Set pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                learning_rate=0.35,
                n_estimators=250
            )
        ))
    ])
   
    # Set parameters for gird search
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.35],
        'clf__estimator__n_estimators': [150, 250]
    }

    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=5, scoring='f1_weighted', verbose=5)

    return cv


def evaluate_model(model, X_test, Y_test, category_name):
    '''
    Was evaluated the model performance using accuracy, precision, recall and f1-score
    '''
    # Predict categories of messages.
    Y_pred = model.predict(X_test)

    # Print accuracy, precision, recall and f1_score for each categories
    for i in range(0, len(category_name)):
        print(category_name[i])
        print("\tAccuracy: {:.3f}\t\t% Precision: {:.3f}\t\t% Recall: {:.3f}\t\t% F1_score: {:.3f}".format(
            accuracy_score(Y_test[:, i], Y_pred[:, i]),
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        ))


def save_model(model, model_filepath):
    '''
    Save the model to a specified path
    Parameters
    model: A Machine learning model
    model_filepath: path where the model will be saved
    
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_name = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_name)

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