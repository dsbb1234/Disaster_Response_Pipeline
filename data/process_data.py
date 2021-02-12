# import the relevant libraries to be used in this ETL Pipeline
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''This function loads the messages and category files as specified in the
    filepaths and merges the both datasets using "id" as the common key between
    messages and categories.

    Inputs:
    messages_filepath - the file containing the messages.  It must be a .csv file format
    categories_filepath - the file containing the categories data.  It must be a .csv format

    Outputs:
    messages - the message file as a DataFrame
    categories - the categories file as a DataFrame
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath, delimiter=',')
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    '''This function splits the categories into separate category columns,
    uses the first row of categories to create column names for the category
    data, and renames the columns of categories with the new column names.
    Duplicate records are removed and category values greater than
    1 are converted to 1 as specified in the requirements.

    Inputs:
    df - the merged dataframe, containing the messages and categories.

    Outputs:
    df - The cleaned df dataframe
    '''
    # create a dataframe of the 36 individual category columns
    # setting expand = True in the split will have the splits expand into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda row: row[:-2]).tolist()

    # assign the cleaned column names to categories
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

        categories['related'] = categories.related.map(lambda x: 1 if x==2 else x)

        # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates()

    # returns the cleaned dataframe for further processing
    return df



def save_data(df, database_filepath):
    '''
    This function saves the cleaned dataframe as an SQLite file using the file name specified by database_filepath

    Inputs:
    df - the merged dataframe, containing the messages and categories.
    database_filepath - the filename and path specified by the user.

    Outputs:
    The cleaned dataframe saved to an SQLite file
    
    '''    
    # save the cleaned file as an SQL database
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql("final_table", engine, index=False, if_exists='replace' )


def main():
    '''
    
    This function executes all of the functions that were created to import,
    clean and save the data as an SQLite file.

    Inputs:
    None

    Outputs:
    Print statements regarding the status of the process until it completes.  
    This includes error messages when the user fails to provide the correct input. 
    
    
    '''
    
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