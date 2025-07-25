'''
next steps:
try to improve the XGB regressor and Random Forest performance
add a ui for users to input their own info
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
import shap 
#use anaconda -> vs code, import scikit learn

import warnings
warnings.filterwarnings('ignore')

def main():
    #get user info
    X_user = get_input()

    #get the dataset 
    df = pd.read_csv('data/full_data.csv')

    print(df.head())
    print("full dataset shape: ", df.shape)
     
    #training and testing split
    features = df.drop(['Calories'], axis=1)
    target = df['Calories'].values
    X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                    test_size=0.2,
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
    
    #just XGB Regressor 
    models[1].fit(X_train, Y_train) #train model
    print(f'{models[1]}: ')
    print("X test shape: " , X_val.shape) # (3000, 8), numpy array

    # make a prediction with the user's info
    #X_user = np.zeros((1, 8))   # create an empty numpy array with dimension (1, 8)
    print("User info: ", X_user)

    # pass user info to XGB model to make predictions
    user_pred = models[1].predict(X_user) 
    print("Prediction for user: ", user_pred)

    #SHAP analysis for most important features in XGB model
    X_val_df = pd.DataFrame(X_val, columns=features.columns.tolist()) 
    # Initialize the SHAP explainer
    explainer = shap.Explainer(models[1])
    # Calculate SHAP values for the test set
    shap_values = explainer(X_val_df)
    shap.plots.bar(shap_values) #Plot feature importance (mean abs shap values)

    '''
    train_preds = models[1].predict(X_train) #make training predictions
    print('Training Error: ', mae(Y_train, train_preds)) #mean absolute error in units of calories.

    val_preds = models[1].predict(X_val) #make testing predictions
    print('Validation Error: ', mae(Y_val, val_preds))
    print()
    '''
   
    #best models in testing: XGB Regressor and Random Forest Regressor 
    '''
    for i in range(5):
        models[i].fit(X_train, Y_train) #train model

        print(f'{models[i]}: ')

        train_preds = models[i].predict(X_train) #make training predictions
        print('Training Error: ', mae(Y_train, train_preds)) #mean absolute error in units of calories.

        val_preds = models[i].predict(X_val) #make testing predictions
        print('Validation Error: ', mae(Y_val, val_preds))
        print()
    '''
    
def get_input():
    #collect user input and convert to the dataset features
    user_input = np.zeros((1, 8))

    gender = float(input("Enter a gender - 0 for male, 1 for female: "))
    age = float(input("Enter an age: "))
    weight = float(input("Enter a weight: "))
    heart_rate = float(input("Enter a heart rate: "))
    body_temp = float(input("Enter a body temp: "))
    duration = float(input("Enter workout duration: "))

    #calculate remaining features
    temp_stress = body_temp * heart_rate
    weight_duration = weight * duration
    stress_effort = temp_stress * (heart_rate * duration)

    #add all the features to the array
    user_input[0] = [gender, age, weight, heart_rate, body_temp, 
                     temp_stress, weight_duration, stress_effort]

    user_list = np.array(user_input)
    print(user_list)
    return user_list

if __name__=="__main__":
    main()


'''
# Correlate features with target. +1 for positive corr, 0 for none, -1 for negative corr
corr = df.corr()
sb.heatmap(corr[['Calories']].sort_values(by='Calories', ascending=False), annot=True)
plt.show()
#most correlated features: Effort, Duration, Temp_Stress, Weight_Duration, Heart_Rate, Body_Temp
'''


'''
#SHAP analysis for most important features in XGB model
#top three features: stress effort, age, weight

X_val_df = pd.DataFrame(X_val, columns=features.columns.tolist()) 
# Initialize the SHAP explainer
explainer = shap.Explainer(models[1])
# Calculate SHAP values for the test set
shap_values = explainer(X_val_df)
shap.plots.bar(shap_values) #Plot feature importance (mean abs shap values)
'''