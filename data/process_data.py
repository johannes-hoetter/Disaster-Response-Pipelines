import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    E-Part of the ETL-Process (Extract):
    Loads the Data from source files into one pandas DataFrame
    :param messages_filepath: path to the messages.csv file
    :param categories_filepath: path to the categories.csv file
    :return: pandas DataFrame containing the two csv Files merged on the ID
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
    T-Part of the ETL-Process (Transform):
    Cleans the Category Column and removes duplicates.
    :param df: pandas DataFrame
    :return: cleaned, duplicate-free pandas DataFrame
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # input df looks like this:
    # index id	categories
    # 0	    2	related-1;request-0;offer-0;aid_related-0; ...
    # 1	    7	related-1;request-0;offer-0;aid_related-1; ...
    # ..    ..  ..
    # To clean the data, the following steps will be executed.
    # 1.) split the values in the categories column on the ; character so that each value becomes a separate column.
    # 2.) the first row of this df can then be used to rename the columns.
    #     it's important to skip the two last characters, as they hold the value of the column.
    # 3.) for each column, take the last character of the value and cast it to int
    #     e.g. 'related-1' gets transformed to 1 (int).
    # 4.) remove the old categories from the original df and replace them with the new, cleaned categories.
    # 5.) drop duplicates.
    # ------------------------------------------------------------------------------------------------------------------
    # 1.)
    categories = df['categories'].str.split(';', expand=True)

    # 2.)
    row = categories.loc[0]
    categories.columns = list(row.apply(lambda x: x[:-2]))

    # 3.)
    for column in categories:
        # set each value to be the last character of the string
        # and convert column from string to numeric
        categories[column] = categories[column].astype(str).apply(lambda x: int(x[-1]))

    # 4.)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # 5.)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename, table_name='Messages'):
    '''
    L-Part of the ETL-Process (Load):
    Saves the given pandas DataFrame in a sqlite Database.
    :param df:
    :param database_filename:
    :param table_name:
    :return: -
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    '''
    Runs the script if the file is called directly.
    :return:  -
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