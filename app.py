# app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('models/engagement_model.pkl', 'rb') as f:
        return pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_data = load_model()
        model = model_data['model']
        
        # Process input data similar to predict.py
        device_mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        features = np.array([[
            data['hour'],
            day_mapping.get(data['day_of_week'], 0),
            device_mapping.get(data['device_category'], 0),
            data.get('engagement_time_msec', 0)
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return jsonify({
            'prediction': bool(prediction),
            'probability': float(probability),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)