# explore_data.py
import pandas as pd
from src.data_cleaning import explore_data  # You'll create this function

# Load the raw data you just saved
df = pd.read_csv('data/raw/test_sample.csv')
explore_data(df)