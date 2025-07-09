import pandas as pd

# Read both datasets
calories_df = pd.read_csv("data/calories.csv")   # has user_id, calories, height, etc.
exercise_df = pd.read_csv("data/exercise.csv")   # has user_id, gender, etc.

# Ensure consistent column formatting
calories_df.columns = calories_df.columns.str.strip()
exercise_df.columns = exercise_df.columns.str.strip()

# Merge side by side on User_ID
merged_df = pd.merge(calories_df, exercise_df, on='User_ID', how='left')

# Print the result
print(merged_df.head())
#save to data folder
merged_df.to_csv("data/full_data.csv", index=False)