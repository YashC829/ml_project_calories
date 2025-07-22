import tkinter as tk
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

def get_input(entry_age, entry_gender, entry_weight, entry_heart, entry_temp, entry_duration, result_label):
    #collect user input and convert to the dataset features
    user_input = np.zeros((1, 8))

    age = entry_age.get()
    gender = entry_gender.get()
    weight = entry_weight.get()
    heart_rate = entry_heart.get()
    body_temp = entry_temp.get()
    duration = entry_duration.get()
    
    #check if field is empty
    if not age or not gender or not weight or not heart_rate or not body_temp or not duration:
        print("Please fill in all fields.")
        result_label.config(text="Please fill in all fields.")
        return
    
    try: #convert input to floats
        age = float(age)
        gender = float(gender)
        weight = float(weight)
        heart_rate = float(heart_rate)
        body_temp = float(body_temp)
        duration = float(duration)

    except ValueError:
        print("Please enter valid numbers in text boxes.")
        result_label.config(text="Please enter valid numbers in text boxes.")
        return
    
    if gender > 1 or gender < 0: # gender input is invalid
        print("Please enter valid number for gender.")
        result_label.config(text="Please enter valid number for gender.")
        return
    
    # check if other inputs are negative
    if age < 0 or weight < 0 or heart_rate < 0 or body_temp < 0 or duration < 0:
        print("Please enter nonnegative numbers in all text boxes.")
        result_label.config(text="Please enter nonnegative numbers in all text boxes.")
        return

    # only print fields if valid inputs 
    print("Gender:", gender)
    print("Age:", age)
    print("Weight:", weight)
    print("Heart Rate:", heart_rate)
    print("Body Temp:", body_temp)
    print("Duration:", duration)

    #calculate remaining features
    temp_stress = body_temp * heart_rate
    weight_duration = weight * duration
    stress_effort = temp_stress * (heart_rate * duration)

    #add all the features to the array
    user_input[0] = [gender, age, weight, heart_rate, body_temp, 
                     temp_stress, weight_duration, stress_effort]

    user_list = np.array(user_input)
    return user_list


def make_prediction(X_user):
     #get the dataset 
    df = pd.read_csv('data/full_data.csv')

    #print(df.head())
    #print("full dataset shape: ", df.shape)
     
    #training and testing split
    features = df.drop(['Calories'], axis=1)
    target = df['Calories'].values
    X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                    test_size=0.2,
                                                    random_state=22) #seed value
    #print("training features shape: ", X_train.shape)
    #print("test features shape: ", X_val.shape) 

    # Normalizing the features for stable and fast training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    #train some ml models and compare their results
    models = [LinearRegression(), XGBRegressor(),
            Lasso(), RandomForestRegressor(), Ridge()]
    
    #just XGB Regressor 
    models[1].fit(X_train, Y_train) #train model
    #print(f'{models[1]}: ')
    #print("X test shape: " , X_val.shape) # (3000, 8), numpy array

    # make a prediction with the user's info
    #X_user = np.zeros((1, 8))   # create an empty numpy array with dimension (1, 8)
    #print("User info: ", X_user)

    # pass user info to XGB model to make predictions
    user_pred = models[1].predict(X_user) 
    print("Calories burned for user: ", user_pred)
    return user_pred[0]
    

def display_window():
    window = tk.Tk()
    window.title("Calories Burned Predictor")  # <-- This sets the window name
    window.geometry("400x400")

    greeting = tk.Label(window, text="Please enter the information", fg="white", bg="black")
    greeting.pack(pady=10)

    # Create a frame to hold labels and entry fields
    form_frame = tk.Frame(window)
    form_frame.pack()

    # Gender row
    tk.Label(form_frame, text="Gender (0 = male, 1 = female):").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    entry_gender = tk.Entry(form_frame, fg="yellow", bg="green", width=10)
    entry_gender.grid(row=0, column=1, padx=10, pady=5)

    # Age row
    tk.Label(form_frame, text="Age:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    entry_age = tk.Entry(form_frame, fg="yellow", bg="blue", width=10)
    entry_age.grid(row=1, column=1, padx=10, pady=5)

    # Weight row
    tk.Label(form_frame, text="Weight:").grid(row=2, column=0, padx=10, pady=5, sticky='e')
    entry_weight = tk.Entry(form_frame, fg="yellow", bg="red", width=10)
    entry_weight.grid(row=2, column=1, padx=10, pady=5)

    # heart rate row
    tk.Label(form_frame, text="Heart Rate:").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    entry_heart = tk.Entry(form_frame, fg="yellow", bg="green", width=10)
    entry_heart.grid(row=3, column=1, padx=10, pady=5)

    # body temp row
    tk.Label(form_frame, text="Body Temp:").grid(row=4, column=0, padx=10, pady=5, sticky='e')
    entry_temp = tk.Entry(form_frame, fg="yellow", bg="blue", width=10)
    entry_temp.grid(row=4, column=1, padx=10, pady=5)

    # duration row
    tk.Label(form_frame, text="Duration:").grid(row=5, column=0, padx=10, pady=5, sticky='e')
    entry_duration = tk.Entry(form_frame, fg="yellow", bg="red", width=10)
    entry_duration.grid(row=5, column=1, padx=10, pady=5)

    # Label to display calories burned or messages
    result_label = tk.Label(window, text="", fg="white", font=("Arial", 16))
    result_label.pack(pady=10)

    # when click submit, get the input and print the resulting list if applicable.
    def on_submit():
        result = get_input(entry_age, entry_gender, entry_weight, entry_heart, entry_temp, entry_duration, result_label)
        #if result is None:
            # Label to display error messages
            #result_label.config(text="Please fill in all fields with valid data.")
        if result is not None:
            print("Saved input:", result)
            calories_burned = make_prediction(result) #use the xgb boost model to make prediction
            print(calories_burned)
            result_label.config(text=f"Calories Burned: {calories_burned:.2f}")
            

    # Submit button
    submit = tk.Button(window, text="Submit!", width=10, height=2, fg="white", bg="black", 
                       command=on_submit)
    submit.pack(pady=15)

    


    window.mainloop()

    #check the result of get_input. if it's not None, then print it out.


def main():
    display_window()

if __name__=="__main__":
    main()
