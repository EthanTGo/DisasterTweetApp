import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = 'id')
    return df



def clean_data(df):
    '''
    Takes the dataframe and perform the required cleaning operations. 
    This includes:
        1. Separating categories columns to 0 or 1
        2. Removing Duplicates
        3. Removing NA/Missing Values
    
    Input: (DataFrame) df. Non-cleaned but merged message and categories dataset
    Output: (DataFrame) df. Cleaned DataFrame after performing operations above
    '''
    # Split Categories into separate categories column
    categories = df[df.columns[4:]]
    categories = categories.categories.str.split(';', expand=True)

    # Selecting the first row
    row = categories.loc[0]
    # Use first row to make column names for each categories column
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert column categories values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x.split('-')[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace= True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # Dropping duplicates
    df.drop_duplicates(inplace = True)
    # Drop NA for only categories column (4th column onwards)
    df = df.dropna(subset = df.columns[4:])
    return df

def save_data(df, database_filename):
    '''
    Save the data into a database using sqlite

    Input: (DataFrame) df. The dataframe that will be saved as a table into a sqlite database
           (String) database_filename. The name of the database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False, if_exists = 'replace')


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