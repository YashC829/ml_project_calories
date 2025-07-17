import tkinter as tk

def get_input():
    age = entry_age.get()
    gender = entry_gender.get()
    weight = entry_weight.get()
    print("Gender:", gender)  # This will print after button click
    print("Age:", age)  # This will print after button click
    print("Weight:", weight)  # This will print after button click

window = tk.Tk()
window.geometry("400x300")  
greeting = tk.Label(text="Please enter the information", 
                    fg="white",
                    bg="black") #label widget
greeting.pack() #add widget to window

entry_gender = tk.Entry(fg="yellow", bg="green", width=20) #entry widget
entry_gender.pack()

entry_age = tk.Entry(fg="yellow", bg="blue", width=20) #entry widget
entry_age.pack()

entry_weight = tk.Entry(fg="yellow", bg="red", width=20) #entry widget
entry_weight.pack()

submit = tk.Button(
    text="Submit!",
    width=10,
    height=3,
    fg="white",
    bg="black",
    command=get_input
) #button widget
submit.pack()

window.mainloop() #keeps window open, listens for events, prevents next code from running


