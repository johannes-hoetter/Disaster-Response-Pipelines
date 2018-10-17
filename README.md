# Disaster-Response-Pipelines
Repository containing Source Code for the "Disaster Response Pipelines" Project from the Udacity Data Science Nanodegree

### Project Motivation
This project is part of the curriculum of the Udacity Data Science Nanodegree. <br>
Aim of this project was to get to know ETL- and ML-Pipelines and how to use them correctly. <br>
The data has been provided by FigureEight.

### Installation
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterMessages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMessages.db RandomForestClassifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to localhost:3001/

### File Descriptions
<p>
|<br>
|- .idea: not important, automatically generated<br>
|- .ipynb_checkpoints: not important, automatically generated<br>
|- app<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- templates<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- go.html: html file to generate the classification results for a given text message<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- master.html: main page (index file)<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- run.py <br>
|- data <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- .ipynb_checkpoints: not important, automatically generated<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- __pycache__: not important, automatically generated<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- DisasterMessages.db: Database containing the data for the model; can be generated through process_data.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- ETL Pipeline Preparation.ipynb: Jupyter Notebook containing the development of process_data.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- categories.csv: File containing the different categories for each text message of the train/test data<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- messages.csv: File containing the different texts and genres for each text message of the train/test data<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- process_data.py: ETL Script, loads data from categories.csv and messages.csv, cleans and merges the data and loads it into a db<br>
|- models<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- .ipynb_checkpoints: not important, automatically generated<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- __pycache__: not important, automatically generated<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- ML Pipeline Preparation.ipynb: Jupyter Notebook containing the development of train_classifier.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- custom_estimators.py: Script containing Custom Scikit-Learn Estimators<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- train_classifier.ipynb: ML Script, builds a model for Classification of Text Messages<br>
</p>

### Note
The work on this project will be continued - by now, the deadline for the project submission has arrived and I'm going
to hand in this version. <br>
Things that are still ToDo:
- Optimize Model Classification Results
- Redesign Visualizations (add more significant ones)
- Change Layout of the Pages (custom made with Bootstrap)
- Rewrite the script files using Django, use Angular or React for Frontend
- Deploy on a Web Server