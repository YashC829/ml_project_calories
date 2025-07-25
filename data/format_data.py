# format the exercise_dataseet_new dataset to fit with the exisiting datasets.

'''
remove ID and other extra columns in new dataset, save as new data

rearrange the order of the columns to fit old dataset
put the new dataset rows underneath the old dataset rows

would like to merge the two datasets but the new dataset doesn't have body temp.
'''

import pandas as pd

# Load the dataset
df = pd.read_csv("data/exercise_dataset_new.csv")

# Drop columns by name
columns_to_drop = ['ID', 'Exercise', 'Dream Weight', 'BMI', 'Weather Conditions', 'Exercise Intensity']  # replace with the actual column names
df = df.drop(columns=columns_to_drop)

# reorder the columns to match the old dataset
new_order = ['Calories Burn', 'Gender', 'Age', 'Actual Weight', 'Heart Rate', 'Duration']  # your desired order
df = df[new_order]

# Save the modified DataFrame to a new CSV
df.to_csv("cleaned_v2.csv", index=False)
