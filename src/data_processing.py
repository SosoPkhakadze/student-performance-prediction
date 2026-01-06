import pandas as pd
import numpy as np
import os

# 1. Load the datasets
df_math = pd.read_csv("data/raw/student-mat.csv", sep=';')
df_por = pd.read_csv("data/raw/student-por.csv", sep=';')

# 2. Add a column to distinguish the subject (Feature Engineering)
df_math['subject'] = 'Math'
df_por['subject'] = 'Portuguese'

# 3. Combine them into one large dataset (Bonus: Multiple Data Sources)
df = pd.concat([df_math, df_por], ignore_index=True)

print(f"Math samples: {df_math.shape[0]}")
print(f"Portuguese samples: {df_por.shape[0]}")
print(f"Total Combined samples: {df.shape[0]}")
df.head()

def clean_and_process_data(df):
    """
    Performs cleaning and feature engineering.
    """
    print("--- Starting Data Cleaning ---")
    
    # 1. Feature Engineering (Bonus Point: New Features)
    # Combine study time and travel time to see total 'academic burden'
    df['total_study_burden'] = df['studytime'] + df['traveltime']
    
    # Create a binary Pass/Fail column (G3 >= 10 is Pass) - Useful for Classification
    df['pass_fail'] = np.where(df['G3'] >= 10, 1, 0)
    
    # 2. Outlier/Anomaly Handling (Page 5)
    # Students with G3=0 usually didn't take the exam. We remove them for better accuracy.
    initial_count = df.shape[0]
    df = df[df['G3'] > 0]
    print(f"Removed {initial_count - df.shape[0]} students with 0 grades (Dropouts/Absent).")
    
    # 3. Encoding Categorical Variables (Page 5)
    # Machine Learning models need numbers, not words like 'GP' or 'F'.
    # We use pd.get_dummies to convert them.
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

if __name__ == "__main__":
    # Apply the cleaning
    df_clean = clean_and_process_data(df)
    
    # 4. Save to Processed Folder (Page 5: Expected Deliverables)
    # Ensure the directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    save_path = "data/processed/cleaned_student_data.csv"
    df_clean.to_csv(save_path, index=False)
    
    print(f"Processing complete. Saved {df_clean.shape[0]} rows to {save_path}")