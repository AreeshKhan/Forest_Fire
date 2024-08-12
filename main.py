# Deploying the app using streamlit

import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Title of the Streamlit app
st.title("Fire Weather Prediction App")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    Temperature = st.sidebar.number_input("Temperature", min_value=-50.0, max_value=50.0, step=0.1)
    RH = st.sidebar.number_input("Relative Humidity", min_value=0.0, max_value=100.0, step=0.1)
    Ws = st.sidebar.number_input("Wind Speed", min_value=0.0, max_value=100.0, step=0.1)
    Rain = st.sidebar.number_input("Rain", min_value=0.0, max_value=500.0, step=0.1)
    FFMC = st.sidebar.number_input("FFMC", min_value=0.0, max_value=100.0, step=0.1)
    DMC = st.sidebar.number_input("DMC", min_value=0.0, max_value=300.0, step=0.1)
    ISI = st.sidebar.number_input("ISI", min_value=0.0, max_value=100.0, step=0.1)
    Classes = st.sidebar.number_input("Classes", min_value=0.0, max_value=1.0, step=0.1)
    Region = st.sidebar.number_input("Region", min_value=0.0, max_value=10.0, step=0.1)
    
    data = {
        'Temperature': Temperature,
        'RH': RH,
        'Ws': Ws,
        'Rain': Rain,
        'FFMC': FFMC,
        'DMC': DMC,
        'ISI': ISI,
        'Classes': Classes,
        'Region': Region
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input features into dataframe
input_df = user_input_features()

# Standardize the input features
input_scaled = standard_scaler.transform(input_df)

# Predict using the Ridge regression model
if st.button("Predict"):
    prediction = ridge_model.predict(input_scaled)
    st.write(f"Predicted Fire Weather Index: {prediction[0]:.2f}")
