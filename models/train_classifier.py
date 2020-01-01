
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
from sklearn.pipeline import FeatureUnion
import pickle
import re
import sys
import warnings
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np
import pandas as pd

from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    The code below load data from database as dataframe
    Input paramaters:
        database_filepath: File path of sql database
    Output:
        X: Message data
        Y: Categories (target)
        category_names: Labels for categories
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='cleanDisasterResponse', con=engine)
    print(df)

    category_name = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_name].values

    return X, y, category_name


def tokenize(text):
    '''
    The code below tokenize the text removing stop words and applying a lemmatize process.
    Input paramaters:
        text: original message text
    Output paramters:
        a clean text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_word)

    return clean_tokens

def build_model():
    '''
    The code below build a machine learning pipeline using tfidf, AdaBoostClassifier and execute a gridsearch to find the best parameters for the model.
    Input paramaters: None
    Output paramaters: Results of GridSearchCV
    
    '''
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
   
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.35],
        'clf__estimator__n_estimators': [150, 250]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, scoring='f1_weighted', verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_name):
    '''
    The code below evaluate the model performance using accuracy, precision, recall and f1-score metrics
    Input paramaters: 
        model: the machine learning model.
        X_test: prediction of test data
        Y_test: true lables for test data
        category_name: Labels for cartegories.
    Output parameters:
        None
    '''
    Y_pred = model.predict(X_test)
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
    Input parameters:
        model: A Machine learning model
        model_filepath: path where the model will be saved
    Output paramters:
         A pickle file of the model
    
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