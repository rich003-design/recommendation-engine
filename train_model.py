# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from pathlib import Path

def load_cleaned_data():
    """Load the cleaned data from the processed directory"""
    data_path = 'data/processed/cleaned_ga4_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned data not found at {data_path}. Run clean_data.py first.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded cleaned data with {len(df)} rows and {len(df.columns)} columns")
    return df

def prepare_features(df):
    """
    Prepare features for the recommendation model
    This creates features that might predict user engagement
    """
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Convert categorical variables to numerical codes
    df_features['device_code'] = df_features['device_category'].astype('category').cat.codes
    df_features['hour_code'] = df_features['hour'].astype('category').cat.codes
    df_features['day_of_week_code'] = df_features['day_of_week'].astype('category').cat.codes
    
    # Select features for the model
    feature_columns = [
        'hour_code',
        'day_of_week_code', 
        'device_code',
        'engagement_time_msec'
    ]
    
    # Target variable: whether user was engaged (1) or not (0)
    # We created this during cleaning with: df['is_engaged'] = df['engagement_time_msec'] > 0
    X = df_features[feature_columns]
    y = df_features['is_engaged'].astype(int)  # Convert True/False to 1/0
    
    return X, y, feature_columns

def train_recommendation_model(X, y):
    """Train a simple classifier to predict user engagement"""
    print("Training recommendation model...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

def save_model(model, feature_columns):
    """Save the trained model and feature information"""
    # Create models directory if it doesn't exist
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'engagement_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_columns': feature_columns,
            'model_type': 'RandomForestClassifier'
        }, f)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def main():
    """Main function to run the training pipeline"""
    try:
        # 1. Load the cleaned data
        print("Step 1: Loading cleaned data...")
        df = load_cleaned_data()
        
        # 2. Prepare features
        print("Step 2: Preparing features...")
        X, y, feature_columns = prepare_features(df)
        print(f"Features: {feature_columns}")
        print(f"Target variable: is_engaged")
        print(f"Data shape: {X.shape}")
        
        # 3. Train the model
        print("Step 3: Training model...")
        model, X_test, y_test = train_recommendation_model(X, y)
        
        # 4. Save the model
        print("Step 4: Saving model...")
        model_path = save_model(model, feature_columns)
        
        print("\n✅ Model training completed successfully!")
        print(f"📁 Model saved at: {model_path}")
        print(f"📊 You can now use this model to predict user engagement")
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())