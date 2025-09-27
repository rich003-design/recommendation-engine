# dashboard.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('GA4 User Engagement Predictor')

# Load model
@st.cache_resource
def load_model():
    with open('models/engagement_model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()

# Input form
st.sidebar.header('User Input')
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
day_of_week = st.sidebar.selectbox('Day of Week', 
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
device = st.sidebar.selectbox('Device', ['mobile', 'desktop', 'tablet'])
engagement_time = st.sidebar.number_input('Engagement Time (ms)', value=30000)

# Prediction
if st.sidebar.button('Predict Engagement'):
    device_mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    features = np.array([[
        hour,
        day_mapping[day_of_week],
        device_mapping[device],
        engagement_time
    ]])
    
    model = model_data['model']
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    st.success(f'Prediction: {"Will Engage" if prediction else "Will Not Engage"}')
    st.info(f'Probability: {probability:.2%}')
    
    if probability > 0.7:
        st.write('✅ High confidence of engagement')
    elif probability > 0.5:
        st.write('⚠️ Medium confidence of engagement')
    else:
        st.write('❌ Low confidence of engagement')