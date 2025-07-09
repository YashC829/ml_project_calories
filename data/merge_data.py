import pandas as pd

# read the datasets
df1 = pd.read_csv(r"calories.csv")
df2 = pd.read_csv(r"exercise.csv")

# print the datasets
print(df1.head())
print(df2.head())
concat_data = pd.concat([df1, df2], ignore_index=True)
print(concat_data)