# Disaster Response Pipeline Project

### Introduction
The main goal of this project is to classify the disaster messages into different categories. This can be done in different ways.
here, i am using the pipeline method to traing the model for dataset. Also, it has a web application script to run the code. 
User can give the message to input box and will get classification results in different categories. The main page of the web app also show
some visualizations of the dataset.

### Files in the repo

*app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
*data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
*models
|- train_classifier.py
|- classifier.pkl # saved model
*README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.d
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
