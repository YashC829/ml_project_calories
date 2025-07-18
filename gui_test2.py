import tkinter as tk
import numpy as np

def get_input():
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
        return
    
    if gender > 1 or gender < 0: # gender input is invalid
        print("Please enter valid number for gender.")
        return
    
    # check if other inputs are negative
    if age < 0 or weight < 0 or heart_rate < 0 or body_temp < 0 or duration < 0:
        print("Please enter nonnegative numbers in all text boxes.")
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
    print(user_list)
    return user_list


window = tk.Tk()
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

# Submit button
submit = tk.Button(window, text="Submit!", width=10, height=2, fg="white", bg="black", command=get_input)
submit.pack(pady=15)

window.mainloop()

