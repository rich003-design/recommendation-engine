# src/data_cleaning.py
import pandas as pd
import numpy as np

def explore_data(df):
    print("=== DATA SHAPE ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n=== FIRST FEW ROWS ===")
    print(df.head())
    
    print("\n=== COLUMN DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())

def clean_ga4_data(df):
    """
    Clean and transform raw GA4 data
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Handle missing values - FIXED
    # First, check the actual dtype of the column
    if df_clean['item_category'].dtype == 'object':
        df_clean['item_category'] = df_clean['item_category'].fillna('Unknown')
    else:
        # If it's not object type, convert it first
        df_clean['item_category'] = df_clean['item_category'].astype(str).fillna('Unknown')
    
    df_clean['engagement_time_msec'] = df_clean['engagement_time_msec'].fillna(0)
    
    # 2. Convert data types
    df_clean['event_datetime'] = pd.to_datetime(df_clean['event_datetime'])
    df_clean['event_date'] = pd.to_datetime(df_clean['event_date'], format='%Y%m%d', errors='coerce')
    
    # 3. Extract useful features from datetime
    df_clean['hour'] = df_clean['event_datetime'].dt.hour
    df_clean['day_of_week'] = df_clean['event_datetime'].dt.day_name()
    
    # 4. Create engagement flags
    df_clean['is_engaged'] = df_clean['engagement_time_msec'] > 0
    
    # 5. Clean categorical data
    df_clean['device_category'] = df_clean['device_category'].str.lower()
    df_clean['traffic_source'] = df_clean['traffic_source'].fillna('Direct')
    
    # 6. Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean