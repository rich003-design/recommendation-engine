# src/feature_utils.py
def get_feature_mappings(df):
    """Get mappings for categorical features"""
    mappings = {}
    
    # Device category mapping
    if 'device_category' in df.columns:
        devices = sorted(df['device_category'].unique())
        mappings['device'] = {code: device for code, device in enumerate(devices)}
    
    # Day of week mapping
    if 'day_of_week' in df.columns:
        days = sorted(df['day_of_week'].unique())
        mappings['day_of_week'] = {code: day for code, day in enumerate(days)}
    
    return mappings

def print_feature_mappings(mappings):
    """Print the feature mappings for reference"""
    print("Feature Mappings:")
    for feature_name, mapping in mappings.items():
        print(f"{feature_name}:")
        for code, value in mapping.items():
            print(f"  {code}: {value}")