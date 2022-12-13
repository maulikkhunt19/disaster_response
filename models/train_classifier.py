import sys
import nltk
import string
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer as wn
from nltk.corpus import stopwords as sw

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['wordnet', 'stopwords'])

def load_data(database_filepath):
    '''
    load_data -> load the data from given database file 
    arg -> database file path
    return -> list of message, values of categories and list of all categories
    '''
    
    sql_engine = create_engine('sqlite:///'+database_filepath)
    disaster_reponse_table = pd.read_sql_table('DisasterResponseTable', sql_engine)
    
    values = disaster_reponse_table['message'].values
    categories = disaster_reponse_table.drop(['id', 'message', 'original', 'genre'], axis = 1)
#     print(categories.shape)
    return values, categories.values, categories.columns

def tokenize(text):
    '''
    tokenize -> dividing the whole text into words and return root form
    arg -> text which we have to tokenize
    return -> a list of the word of given message in the root forms
    '''
    str_translator = str.maketrans('', '', string.punctuation)
    text_edited = text.lower().strip().translate(str_translator)
    
    text_tokenizer = word_tokenize(text_edited)
    stopwords = sw.words('english')
    
    wordnet_lemmatizer = wn()
    words = [wordnet_lemmatizer.lemmatize(word) for word in text_tokenizer if word not in stopwords]
#     print(words)
    return words
    
def build_model():
    '''
    build_model -> building a model to classify the message with pipeline
    return -> return classification model
    '''
    rf_estimator = RandomForestClassifier()
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_transformer', TfidfTransformer()),
        ('multi_classifier', MultiOutputClassifier(estimator=rf_estimator))
    ])
#     to reduce the .pkl file size which was around 2GB for this 
#     parameters = {
#         'tfidf_transformer__use_idf': [True, False],
#         'multi_classifier__estimator__n_estimators': [50, 100, 150]
#     }
    
    parameters = {
        'tfidf_transformer__use_idf': [True, False],
        'multi_classifier__estimator__n_estimators': [20]
    }
  
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model ->  to predict the trained model by giving the classification report
    arg -> trained model, test data with actual output and list of categories
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    save_model -> to save the model to .pkl file
    arg -> trained model and output file path
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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