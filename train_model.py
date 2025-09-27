# train_model.py (KFP Component Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from kfp.dsl import component, Input, Output, Model, Dataset, Metrics, HTML

@component
def train_model_component(
    cleaned_data: Input[Dataset],
    trained_model: Output[Model],
    model_metrics: Output[Metrics],
    feature_report: Output[HTML]
) -> str:
    """
    KFP component for training a recommendation model on cleaned GA4 data.
    """
    
    # 1. Load the cleaned data from KFP-provided path
    print(f"📊 Loading cleaned data from: {cleaned_data.path}")
    df = pd.read_csv(cleaned_data.path)
    print(f"Data shape: {df.shape}")
    
    # 2. Prepare features (your existing function)
    def prepare_features(df):
        """Prepare features for the recommendation model"""
        df_features = df.copy()
        
        # Convert categorical variables to numerical codes
        df_features['device_code'] = df_features['device_category'].astype('category').cat.codes
        df_features['hour_code'] = df_features['hour'].astype('category').cat.codes
        df_features['day_of_week_code'] = df_features['day_of_week'].astype('category').cat.codes
        
        # Select features for the model
        feature_columns = [
            'hour_code', 'day_of_week_code', 'device_code', 'engagement_time_msec'
        ]
        
        # Target variable: whether user was engaged (1) or not (0)
        X = df_features[feature_columns]
        y = df_features['is_engaged'].astype(int)
        
        return X, y, feature_columns
    
    # 3. Train the model (your existing function)
    def train_recommendation_model(X, y):
        """Train a classifier to predict user engagement"""
        print("🤖 Training recommendation model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with accuracy: {accuracy:.2f}")
        
        return model, X_test, y_test, accuracy
    
    # 4. Execute the training pipeline
    X, y, feature_columns = prepare_features(df)
    model, X_test, y_test, accuracy = train_recommendation_model(X, y)
    
    # 5. Save the model to KFP-provided path
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy
    }
    
    with open(trained_model.path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved to: {trained_model.path}")
    
    # 6. Log metrics to KFP UI
    model_metrics.log_metric("accuracy", float(accuracy))
    model_metrics.log_metric("n_features", len(feature_columns))
    model_metrics.log_metric("n_samples", len(df))
    
    # 7. Generate HTML report
    report_content = f"""
    <html>
    <body>
        <h2>Model Training Report</h2>
        <p><strong>Accuracy:</strong> {accuracy:.3f}</p>
        <p><strong>Features:</strong> {', '.join(feature_columns)}</p>
        <p><strong>Training Samples:</strong> {len(df):,}</p>
        <p><strong>Model Type:</strong> Random Forest Classifier</p>
        <h3>Classification Report:</h3>
        <pre>{classification_report(y_test, model.predict(X_test))}</pre>
    </body>
    </html>
    """
    
    with open(feature_report.path, 'w') as f:
        f.write(report_content)
    
    return f"Model training completed! Accuracy: {accuracy:.3f}, Model saved to: {trained_model.path}"

# Keep your original functions for local testing (optional)
def load_cleaned_data_local():
    """Local version for testing without KFP"""
    data_path = 'data/processed/cleaned_ga4_data.csv'
    df = pd.read_csv(data_path)
    return df

def main_local():
    """Local testing version"""
    df = load_cleaned_data_local()
    X, y, feature_columns = prepare_features(df)
    model, X_test, y_test, accuracy = train_recommendation_model(X, y)
    
    # Save locally
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'accuracy': accuracy
    }
    
    with open('models/engagement_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Local model saved with accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    # Run local version when executed directly
    main_local()