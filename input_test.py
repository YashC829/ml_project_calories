#get a float from user input
#run with terminal: python input_test.py

#input we want: gender, age, weight, heart rate, body temp, duration. need the units for last 4

import numpy as np

#list of the data
user_input = []


num = float(input("Enter a gender - 0 for male, 1 for female: "))
print(num)
user_input.append(num)

num = float(input("Enter an age: "))
print(num)
user_input.append(num)

num = float(input("Enter a weight: "))
print(num)
user_input.append(num)

num = float(input("Enter a heart rate: "))
print(num)
user_input.append(num)


num = float(input("Enter a body temp: "))
print(num)
user_input.append(num)

num = float(input("Enter workout duration: "))
print(num)
user_input.append(num)

user_list = np.array(user_input)
print(user_list)