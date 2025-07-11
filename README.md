# ml_project_calories
Predict the calories that someone burnt during a workout, based on biological and workout data.
Created by Yash Chadda.

Sources:
1. https://www.geeksforgeeks.org/machine-learning/calories-burnt-prediction-using-machine-learning/
2. https://www.geeksforgeeks.org/python/how-to-join-datasets-with-same-columns-and-select-one-using-pandas/

Datasets:
https://www.kaggle.com/datasets/sparkyxt/calories-burning-dataset?resource=download
The datasets are Calories.csv and Exercise.csv. They are referenced in the first source and were
originally posted on Kaggle.

Model: 
I used the XGBoost model (from xgboost library) to train and make predictions. Currently, the model has an absolute mean error of 1.22 for test data.
