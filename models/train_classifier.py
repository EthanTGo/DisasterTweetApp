'''
The following code was adopted from the notebook provided. Most of the functions were taken there.
The Notebook shows examples on what each function does. 
'''
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath, table_name = 'df', column_name = message):
    '''
    Function that is used to load the data from a database

    Input: (String) database_filepath. The path of the database
    Output: X = The input variable for the model. Default is message
            Y = The output label we are trying to predict. It looks from the 4th columns onwards 
            due to the nature of the data
            category_names = names of the categories
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df[column_name].values
    Y = df.iloc[:, 4:].values
    categories = Y.columns.tolist()
    return X,Y,categories

def tokenize(text):
    '''
    Tokenize the text, lemmatize it and cleaning it and lower casing and stripping any whitespaces

    Input: (Array of String) text. This is the text that needs to be inputted for futher cleaning
    Output: (Array) clean_tokens. This is the array fo clean tokens after being tokenized
    '''
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in token:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
​

def build_model():
    '''
    Create the pipeline model. By default there is a CountVectorizer + TfidfTransformer and MultioutputClassifier

    Output: (Pipeline) A pipeline object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model for each of the category

    Input: (Pipeline/Estimator) model. The model that was used on the training dataset
           (DataFrame/Array) X_test. The testing dataset's input 
           (DataFrame/Array) Y_test. The testing dataset's output
           (Array) category_names. A list of category to iterate over for evaluation (must be in the same order as Y result)
    '''
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    y_test = pd.DataFrame(Y_test, columns = category_names)

    # Iterate over each category and output the F1 score, confusion matrix through the classification_report method
    for i, var in enumerate(category_names):
        print("For label " + var + " this is the result.")
        print(classification_report(y_test.iloc[:,i], 
                                    y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    '''
    Used to save the model into a pickle file

    Input: (Pipeline/Estimator) model. The model that we want to be saved
           (String) model_filepath. The filepath of where the model's pickle file should be saved
    '''
    pickle.dump(model,open(model_filepath,'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
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