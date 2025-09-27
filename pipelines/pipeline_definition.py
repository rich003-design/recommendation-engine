# pipelines/pipeline_definition.py
from kfp import dsl

# Import your KFP components
from clean_data import clean_ga4_data_component
from train_model import train_model_component
from predict import predict_component

@dsl.pipeline(
    name="ml-training-pipeline",
    description="End-to-end ML pipeline for GA4 data"
)
def ml_pipeline(
    data_path: str = "data/raw/test_sample.csv"
):
    # Step 1: Clean the data
    clean_data_op = clean_ga4_data_component(
        input_data_path=data_path
    )
    
    # Step 2: Train model using the cleaned data
    train_model_op = train_model_component(
        cleaned_data=clean_data_op.outputs['cleaned_data']
    )
    
    # Step 3: Make predictions
    predict_op = predict_component(
        model=train_model_op.outputs['trained_model'],  # Fixed: 'trained_model' not 'model'
        data=clean_data_op.outputs['cleaned_data']
    )