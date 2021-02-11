# Disaster Response Pipelines

### Description
This project creates a machine learning pipeline to categorize messages related to disaster events by category.  This allows an emergency worker to know which disaster relief agencies to send  the messages for action.  The project includes an ETL Pipeline, an ML Pipeline, and a web app to input a disaster message and display the categories that are appropriate for receiving the messages.

### Extract, Transform and Load (ETL) Pipeline
The first part of your data pipeline is the Extract, Transform, and Load process. Here, you will read the dataset, clean the data, and then store it in a SQLite database.  Data cleaning is performed using Jupyter Notebook and Pandas.  After transformation, the data is loaded into an SQLite database using SQLAlchemy.

Here is a summary of the ETL Discovery process

Data was imported (extracted) using read_csv.  The contents of both the messages.csv and categories.csv files were reviewed to understand the data contained in each dataframe.  Since both dataframes had "id" as a common variable, the dataframes were merged using "id".  

To support the Machine Learning pipeline, the categories were transformed into columns, assigned category labels and the text was removed from the merged dataframe. Duplicate messages were found and removed from the merged dataframe.

The categories of messages were explored using df.sum() and it was discovered there are some categories with much larger numbers of messages, indicating this dataframe is likely to result in an unbalanced model.  The cleaned dataframe was saved to an SQLite database, using SQLAlchemy.

### Machine Learning (ML) Pipeline
In this portion of the project, data was:
      1) Imported from an SQLite database, <br>
      2) Tokenized (split into individual words and special characters/extra spaces removed),<br>
      3) Lemmatized, split into test and training data and loaded into a machine learning model using a custom function. 
      4) Pipeline and GridSearchCV were used in order to incorporate multiple machine learning libraries and to optimize parameters used in the model.<br>
      5) The data was split into test and train datasets and 80% of the data was used to train the model.<br>
      6) The model was build using the build_model() function<br>
      7) The model was fit to the learning data<br>
      8) The model was fit to the test data to support testing the model.<br>
      9) Classification_report was used to display results of testing.  Precision, recall and F1 score were calculated<br>
      10) Parameters were modified to optimize the model.  Additional models were tried and results were evaluated,<br>
      11) The model with the highest precision, recall and F1 score was chosen as the final model used in the python<br> train_classifier.py<br>
      12) The model was saved as a pickle file<br>

The final model uses the catgories and messages supplied by the user, trains on the data and applies the trained model to identify new data entered via a web_page.

### Observations and Conclusion

This application successfully demonstrates the ability to create the Extract Transform Load (ETL) and Machine Learning (ML) pipelines.  In the ETL phase, data was loaded, cleaned, transformed and loaded into an SQLite database in preparation for an ML pipeline.  In the ML phase, SQLite data was loaded, tokenized, and lemmatized in preparation for use in the machine learning model.  Machine learning models were created, tuned and scored, in an attempt to create a model having the best precision, recall and F1 score.  Finally, the model was saved as a pickle file to be used by the run.py program.

There were some issues with this model that should be adressed in future models:
      1) It was noted in the ETL phase that the data appeared unbalanced.  Certain categories contained much more data than other categories.  In the machine learning model, those categories would receive much higher weighting than the categories containing less data.  Lesser weigted categories are ignored by the model.<br>
      Recommendation - The next iteration of this model should modify the test/train/split data to create a more balanced dataset.  Possibly explore using other functions such as imbalanced-learn.org's Multiclass "make_imbalance" to apply to the train/testing data prior to train/test/split.
      2) The application could also include automation to automatically send the messages to the appropriate agencies, avoiding user intervention.

### Files in the Repository

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # contains the message id and the categories the message was assigned to
|- disaster_messages.csv  # contains the message id, message, original message (before translation) and the message genre
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py - the Machine Learning Pipeline code
|- DisasterResponse.db - this is the output of the ETL pipeline (an SQLite database) containing the cleaned message information (merged messages and categories)
|- classifier.pkl  # saved model as a pickle file
|- starting_verb_extractor.py - contains the tokenize and StartingVerbExtractor functions

- Snapshots
|- pict_process_data.png - snapshot showing successful execution of the process_data.py program
|- pict_train_classifier_py.png - snapshot executing the train_classifier.py program
|- pict_train_classifier_py2.png - snapshot showing successful completion of the train_classifier.py program.
|- pict_run.png - snapshot executing the run.py program
|- pict_run_py2.png - snapshot showing the train_classifier.py program completing and saving the training model
|- pic_run_py_for_web6.png - snapshot showing the program taking the input from the web page for classification.
|- pic_web1.png - Header for the web page, allowing the submission of a message for categorization
|- pic_web2.png - First chart: Message Genre Distribution, supplied by Udacity
|- pic_web3.png - Second chart: Message Category Distribution Bar Chart
|- pic_web4.png - Third chart:  Percentage of Direct/Indirectly Reported Messages
|- pic_web5.png - Example of a message that was submitted to the web page for classification
|- pic_web6.png - Result of classification, showing the message was "Related", "Aid Related", "Other Aid", and "Storm"

### Program Execution

The program can be executed by running the following commands from the terminal:<br>
1. Run the following commands in the project's root directory to set up your database and model.<br>

    - To run ETL pipeline that cleans data and stores in database<br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`<br>
    - To run ML pipeline that trains classifier and saves<br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`<br>

2. Run the following command in the app's directory to run your web app.<br>
    `python run.py`<br>

3. Go to http://0.0.0.0:3001/<br>

Udacity provided a Flask app to allow display of results, in the run.py python program.<br>
