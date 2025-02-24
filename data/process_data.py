import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    The code below load message and categories data
    Input parameters:
        messages_filepath: File path of messages data
        categories_filepath: File path of categories data
    Output parameters:
        df: A merged dataset from messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer') 
    return df


def clean_data(df):
    '''
    The code below clean the dataset
    Input parameters:
        df: The merged dataset (messages and categories)
    Output parameters:
        df: Cleaned dataset
    '''
    categories = df['categories'].str.split(";",expand = True)
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
  
    categories = (categories > 0).astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace = True)
    return df


def save_data(df, database_filename):
    '''
    The code below save the dataframe into sqlite database
    Input parameters:
        df: cleaned dataset
        database_filename: database name, e.g. DisasterMessages.db
    Output parameters: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleanDisasterResponse', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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