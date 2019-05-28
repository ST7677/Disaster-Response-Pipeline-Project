# Disaster-Response-Pipeline-Project
Udacity Data Science Nano Degree - Project 5

## Project Motivation

In this project, we use Data Engineering skills to analyze disaster response data to build a model for an API that classifies disaster messages.
Figure Eight has provided thousands of messages that have been sorted into 36 categories. This will help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources. The ML model is more accurate than simple keyword search.

## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file to run the browser app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset of all the categories  
    │   ├── disaster_messages.csv            # Dataset of all the messages
    │   └── process_data.py                  # Module for Data cleaning
    ├── models
    │   └── train_classifier.py              # Training ML model           
    └── README.md

### Instructions: 
1. Run the following commands in the project's root directory to set up your database and model.
    (If DisasterResponse.db and classifier.pkl exist already the skip this step and got to step 2 below)

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![ScreenShot](Project5Screenshot1.png)
![ScreenShot](Project5Screenshot2.png)
![ScreenShot](Project5Screenshot3.png)
![ScreenShot](Project5Screenshot4.png)
