# predict.py (KFP Component Version)
import pandas as pd
import pickle
import numpy as np
import json
from kfp.dsl import component, Input, Output, Model, Dataset, Metrics, Markdown

@component
def predict_component(
    model: Input[Model],
    data: Input[Dataset],
    predictions: Output[Dataset],
    prediction_metrics: Output[Metrics],
    prediction_report: Output[Markdown]
) -> str:
    """
    KFP component for making predictions using a trained model.
    """
    
    # 1. Load the trained model from KFP-provided path
    print(f"🤖 Loading model from: {model.path}")
    with open(model.path, 'rb') as f:
        model_data = pickle.load(f)
    
    trained_model = model_data['model']
    feature_columns = model_data['feature_columns']
    model_accuracy = model_data.get('accuracy', 'Unknown')
    
    print(f"✅ Model loaded. Features: {feature_columns}")
    print(f"📊 Model accuracy: {model_accuracy}")
    
    # 2. Load the data for prediction from KFP-provided path
    print(f"📈 Loading data for prediction from: {data.path}")
    df = pd.read_csv(data.path)
    print(f"📋 Data shape for prediction: {df.shape}")
    
    # 3. Prepare features for prediction (similar to training)
    def prepare_features_for_prediction(df):
        """Prepare features for prediction using the same logic as training"""
        df_features = df.copy()
        
        # Convert categorical variables to numerical codes (same as training)
        df_features['device_code'] = df_features['device_category'].astype('category').cat.codes
        df_features['hour_code'] = df_features['hour'].astype('category').cat.codes
        df_features['day_of_week_code'] = df_features['day_of_week'].astype('category').cat.codes
        
        # Use the same feature columns that the model was trained on
        X_pred = df_features[feature_columns]
        
        return X_pred, df_features
    
    # 4. Make predictions
    X_pred, original_df = prepare_features_for_prediction(df)
    
    print("🎯 Making predictions...")
    predictions_array = trained_model.predict(X_pred)
    probabilities = trained_model.predict_proba(X_pred)[:, 1]  # Probability of class 1
    
    # 5. Add predictions to the original dataframe
    results_df = original_df.copy()
    results_df['engagement_prediction'] = predictions_array
    results_df['engagement_probability'] = probabilities
    results_df['prediction_label'] = results_df['engagement_prediction'].map({0: 'Not Engaged', 1: 'Engaged'})
    
    # 6. Save predictions to KFP-provided output path
    results_df.to_csv(predictions.path, index=False)
    print(f"💾 Predictions saved to: {predictions.path}")
    
    # 7. Calculate prediction metrics
    n_predictions = len(results_df)
    engagement_rate = results_df['engagement_prediction'].mean()
    avg_probability = results_df['engagement_probability'].mean()
    
    # 8. Log metrics to KFP UI
    prediction_metrics.log_metric("n_predictions", n_predictions)
    prediction_metrics.log_metric("predicted_engagement_rate", float(engagement_rate))
    prediction_metrics.log_metric("average_confidence", float(avg_probability))
    prediction_metrics.log_metric("model_accuracy", float(model_accuracy) if model_accuracy != 'Unknown' else 0)
    
    # 9. Generate prediction report
    report_content = f"""
    # Prediction Results Report
    
    ## Summary
    - **Total Predictions Made**: {n_predictions:,}
    - **Predicted Engagement Rate**: {engagement_rate:.1%}
    - **Average Confidence**: {avg_probability:.3f}
    - **Model Accuracy**: {model_accuracy if model_accuracy != 'Unknown' else 'N/A'}
    
    ## Prediction Distribution
    - Engaged Users: {results_df['engagement_prediction'].sum():,}
    - Not Engaged Users: {(results_df['engagement_prediction'] == 0).sum():,}
    
    ## Top 10 Predictions Sample
    {results_df[['hour', 'day_of_week', 'device_category', 'engagement_probability', 'prediction_label']].head(10).to_markdown(index=False)}
    
    ## Feature Importance
    The model was trained using these features: {', '.join(feature_columns)}
    """
    
    with open(prediction_report.path, 'w') as f:
        f.write(report_content)
    
    # 10. Print some sample predictions
    print("\n🔍 Sample Predictions:")
    sample_results = results_df[['hour', 'day_of_week', 'device_category', 'engagement_probability', 'prediction_label']].head(5)
    print(sample_results.to_string(index=False))
    
    return f"Prediction completed! Made {n_predictions} predictions. Engagement rate: {engagement_rate:.1%}"

# Keep your original EngagementPredictor class for local/testing use
class EngagementPredictor:
    """Original predictor for local/testing use (outside KFP)"""
    
    def __init__(self, model_path='models/engagement_model.pkl'):
        self.model_path = model_path
        self.local_model = None
        self.feature_columns = None
        self._load_local_model()
    
    def _load_local_model(self):
        """Load the local model"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.local_model = model_data['model']
        self.feature_columns = model_data['feature_columns']
    
    def predict_single(self, user_data):
        """Predict engagement for a single user"""
        device_mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        features = np.array([[
            user_data['hour'],
            day_mapping.get(user_data['day_of_week'], 0),
            device_mapping.get(user_data['device_category'], 0),
            user_data['engagement_time_msec']
        ]])
        
        prediction = self.local_model.predict(features)[0]
        probability = self.local_model.predict_proba(features)[0][1]
        
        return {
            'will_engage': bool(prediction),
            'probability': float(probability),
            'source': 'local_model'
        }

def main_local():
    """Local testing version (without KFP)"""
    # Load sample data and make predictions
    data_path = 'data/processed/cleaned_ga4_data.csv'
    model_path = 'models/engagement_model.pkl'
    
    df = pd.read_csv(data_path)
    predictor = EngagementPredictor(model_path)
    
    # Test with first row
    sample_user = df.iloc[0].to_dict()
    result = predictor.predict_single(sample_user)
    
    print("🔮 Single Prediction Test:")
    print(f"User: {sample_user}")
    print(f"Prediction: {result}")

if __name__ == "__main__":
    # Run local version when executed directly
    main_local()