# Data Engineering
# Project: Disaster-Response-Pipeline
1. Project Overview
This project consists in build a web application where will execute two specific task:  A ETL process and build a machine learning Pipeline. As output the web application will display visualizations of the data.

2. Description Project
There are three main folders:

<pre><code> 2.1 data
  The data folder contains the following files:
 * disaster_categories.csv: dataset with all the categories
 * disaster_messages.csv: dataset with all the messages
 * process_data.py: ETL pipeline scripts to read csv format, clean the text, and save the data into a SQLite database.
 * DisasterResponse.db: output of the ETL pipeline.

2.2 models
The models folder contains the following files:
 * train_classifier.py: machine learning pipeline scripts to build, evaluate and export a classifier
 * classifier.pkl: output of the machine learning pipeline.
 
2.3 app
The app folder contains the following files:
 * run.py: Flask file to run the web application
 * templates: html file for the web applicatin
 </code></pre>

3. Runing
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

4. Required libraries
* pandas
* nltk
* numpy
* pandas
* scikit-learn
* sqlalchemy 

5. Acknowledgements

I wish to thank Figure Eight for dataset.




