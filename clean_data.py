# clean_data.py
import pandas as pd
from kfp.dsl import component, Output, Dataset
from src.data_cleaning import clean_ga4_data

@component
def clean_ga4_data_component(
    input_data_path: str,           # You provide the input path
    cleaned_data: Output[Dataset]   # KFP provides the output path automatically
) -> str:
    """
    KFP component for cleaning GA4 data.
    
    Args:
        input_data_path: Path to the raw CSV data file
        cleaned_data: KFP-managed output dataset
    
    Returns:
        Success message with input/output paths
    """
    
    # Load raw data from the provided input path
    raw_df = pd.read_csv(input_data_path)
    print(f"✅ Loaded raw data from: {input_data_path}")
    print(f"📊 Raw data shape: {raw_df.shape}")
    
    # Clean the data using your existing function
    cleaned_df = clean_ga4_data(raw_df)
    print(f"✨ Data cleaning completed")
    print(f"📈 Cleaned data shape: {cleaned_df.shape}")
    
    # Save cleaned data to the KFP-provided output path
    # NO need for manual directory creation - KFP handles this!
    cleaned_df.to_csv(cleaned_data.path, index=False)
    
    print(f"💾 Cleaned data saved to: {cleaned_data.path}")
    
    return f"Data cleaning completed! Input: {input_data_path}, Output: {cleaned_data.path}"