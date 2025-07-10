'''
next steps:
try to improve the XGB regressor and Random Forest performance
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
#use anaconda -> vs code, import scikit learn

import warnings
warnings.filterwarnings('ignore')

def main():
    #get the dataset 
    df = pd.read_csv('data/full_data.csv')

    #remove weight and duration columns (highly correlated with calories, avoid data leakage)
    to_remove = ['Weight', 'Duration']
    df.drop(to_remove, axis=1, inplace=True)

    print(df.head())
    print("full dataset shape: ", df.shape)

    #training and testing split
    features = df.drop(['User_ID', 'Calories'], axis=1)
    target = df['Calories'].values

    X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                    test_size=0.1,
                                                    random_state=22) #seed value
    print("training features shape: ", X_train.shape)
    print("test features shape: ", X_val.shape) 

    # Normalizing the features for stable and fast training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    #train some ml models and compare their results
    models = [LinearRegression(), XGBRegressor(),
            Lasso(), RandomForestRegressor(), Ridge()]

    #best models in testing: XGB Regressor and Random Forest Regressor 
    for i in range(5):
        models[i].fit(X_train, Y_train) #train model

        print(f'{models[i]}: ')

        train_preds = models[i].predict(X_train) #make training predictions
        print('Training Error: ', mae(Y_train, train_preds)) #mean absolute error in units of calories.

        val_preds = models[i].predict(X_val) #make testing predictions
        print('Validation Error: ', mae(Y_val, val_preds))
        print()

if __name__=="__main__":
    main()