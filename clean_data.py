import os
import pandas as pd
from src.data_cleaning import clean_ga4_data

# Load raw data
raw_df = pd.read_csv('data/raw/test_sample.csv')

# Clean the data
cleaned_df = clean_ga4_data(raw_df)

# Ensure the directory exists before saving
output_path = 'data/processed/cleaned_ga4_data.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # This creates all folders in the path

# Save cleaned data
cleaned_df.to_csv(output_path, index=False)
print("Data cleaning completed! Saved to", output_path)