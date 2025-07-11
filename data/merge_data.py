import pandas as pd

# Read both datasets
calories_df = pd.read_csv("data/calories.csv")   # has user_id, calories, height, etc.
exercise_df = pd.read_csv("data/exercise.csv")   # has user_id, gender, etc.

# Ensure consistent column formatting
calories_df.columns = calories_df.columns.str.strip()
exercise_df.columns = exercise_df.columns.str.strip()

# Merge side by side on User_ID
merged_df = pd.merge(calories_df, exercise_df, on='User_ID', how='left')

# change all male and female strings to 0 and 1 respectively
merged_df['Gender'] = merged_df['Gender'].map({'male': 0, 'female': 1})

# Create interaction features
# Avoid division by zero or nulls
merged_df['BMI'] = merged_df.apply(
    lambda row: row['Weight'] / ((row['Height'] / 100) ** 2) if row['Height'] > 0 else None,
    axis=1
)
merged_df['Effort'] = merged_df['Heart_Rate'] * merged_df['Duration']
merged_df['Temp_Stress'] = merged_df['Body_Temp'] * merged_df['Heart_Rate']
merged_df['Weight_Duration'] = merged_df['Weight'] * merged_df['Duration']
merged_df['Stress_Effort'] = merged_df['Temp_Stress'] * merged_df['Effort']
merged_df['Effort_Squared'] = merged_df['Effort'] ** 2

# Round selected columns
merged_df['BMI'] = merged_df['BMI'].round(2)
merged_df['Temp_Stress'] = merged_df['Temp_Stress'].round(2)
merged_df['Stress_Effort'] = merged_df['Stress_Effort'].round(2)

# Removing least important features.
to_remove = ['User_ID', 'Height', 'Duration', 'Effort_Squared', 'BMI', 'Effort']
merged_df.drop(to_remove, axis=1, inplace=True)

# Print the result
print(merged_df.head())
#save to data folder
merged_df.to_csv("data/full_data.csv", index=False)