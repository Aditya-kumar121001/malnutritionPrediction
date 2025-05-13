import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Load the dataset
data = pd.read_csv("dataset_forecast_monthly_filled.csv")

# Step 1: Encode the target variables into numeric values
# Encode UNDERWEIGHT
underweight_mapping = {'normal': 0, 'moderately underweight': 1, 'severely underweight': 2}
data['UNDERWEIGHT'] = data['UNDERWEIGHT'].apply(lambda x: 'normal' if 'normal' in str(x) else ('moderately underweight' if 'moderately underweight' in str(x) else 'severely underweight'))
data['UNDERWEIGHT'] = data['UNDERWEIGHT'].map(underweight_mapping)
data['UNDERWEIGHT'] = data['UNDERWEIGHT'].fillna(0)

# Encode STUNTED
stunted_mapping = {'normal': 0, 'moderately stunted': 1, 'severely stunted': 2}
data['STUNTED'] = data['STUNTED'].apply(lambda x: 'normal' if 'normal' in str(x) else ('moderately stunted' if 'moderately stunted' in str(x) else 'severely stunted'))
data['STUNTED'] = data['STUNTED'].map(stunted_mapping)
data['STUNTED'] = data['STUNTED'].fillna(0)

# Encode WASTED
wasted_mapping = {'normal': 0, 'mam': 1, 'sam': 2}
data['WASTED'] = data['WASTED'].apply(lambda x: 'normal' if 'normal' in str(x) else ('mam' if 'mam' in str(x) else 'sam'))
data['WASTED'] = data['WASTED'].map(wasted_mapping)
data['WASTED'] = data['WASTED'].fillna(0)

# Step 2: Compute target encodings for SECTOR_code and Awc_code_new for each target
underweight_sector_encoding = data.groupby('SECTOR_code')['UNDERWEIGHT'].mean()
underweight_awc_encoding = data.groupby('Awc_code_new')['UNDERWEIGHT'].mean()
stunted_sector_encoding = data.groupby('SECTOR_code')['STUNTED'].mean()
stunted_awc_encoding = data.groupby('Awc_code_new')['STUNTED'].mean()
wasted_sector_encoding = data.groupby('SECTOR_code')['WASTED'].mean()
wasted_awc_encoding = data.groupby('Awc_code_new')['WASTED'].mean()


# Load the saved models and scaler
underweight_model = load_model("lstm_underweight_model.h5")
stunted_model = load_model("lstm_stunted.h5")
wasted_model = load_model("lstm_wasted.h5")
scaler = joblib.load("scaler.pkl")

# Feature columns expected by the models
feature_cols_underweight = [
    'WEIGHT', 'HEIGHT', 'Age_in_month', 'GENDER',
    'BENEFICIARY_children_0m_6m', 'BENEFICIARY_children_6m_3y', 'BENEFICIARY_children_3y_6y',
    'SECTOR_code_encoded_underweight', 'Awc_code_new_encoded_underweight'
]
feature_cols_stunted = [
    'WEIGHT', 'HEIGHT', 'Age_in_month', 'GENDER',
    'BENEFICIARY_children_0m_6m', 'BENEFICIARY_children_6m_3y', 'BENEFICIARY_children_3y_6y',
    'SECTOR_code_encoded_stunted', 'Awc_code_new_encoded_stunted'
]
feature_cols_wasted = [
    'WEIGHT', 'HEIGHT', 'Age_in_month', 'GENDER',
    'BENEFICIARY_children_0m_6m', 'BENEFICIARY_children_6m_3y', 'BENEFICIARY_children_3y_6y',
    'SECTOR_code_encoded_wasted', 'Awc_code_new_encoded_wasted'
]

# Streamlit app
st.title("Child UNDERWEIGHT, STUNTED, and WASTED Prediction App")
st.write("Enter the data for a child over 3 time points to predict UNDERWEIGHT, STUNTED, and WASTED status.")

# Create a form for user input with default test case values
with st.form("child_data_form"):
    st.write("### Child Data Input")
    
    # Input fields for 3 time points with default values
    child_data = []
    # Default values for the test case
    default_weights = [7.0, 7.1, 7.2]  # kg
    default_heights = [75.0, 75.5, 76.0]  # cm
    default_ages = [24, 25, 26]  # months
    default_gender = "F"
    default_beneficiary_type = "children_6m_3y"
    default_sector_code = "SECTOR_1"
    default_awc_code = "AWC_1"

    for i in range(1, 4):
        st.write(f"#### Time Point {i}")
        weight = st.number_input(f"Weight (kg) at Time Point {i}", min_value=0.0, value=default_weights[i-1], step=0.1)
        height = st.number_input(f"Height (cm) at Time Point {i}", min_value=0.0, value=default_heights[i-1], step=0.1)
        age = st.number_input(f"Age (months) at Time Point {i}", min_value=0, value=default_ages[i-1], step=1)
        gender = st.selectbox(f"Gender at Time Point {i}", options=['M', 'F'], index=1 if default_gender == "F" else 0)
        beneficiary_type = st.selectbox(f"Beneficiary Type at Time Point {i}", 
                                     options=['children_0m_6m', 'children_6m_3y', 'children_3y_6y'],
                                     index=1 if default_beneficiary_type == "children_6m_3y" else 0)
        sector_code = st.text_input(f"SECTOR_code at Time Point {i}", value=default_sector_code)
        awc_code = st.text_input(f"Awc_code_new at Time Point {i}", value=default_awc_code)
        
        # Create a dictionary for this time point
        time_point_data = {
            'WEIGHT': weight,
            'HEIGHT': height,
            'Age_in_month': age,
            'GENDER': gender,
            'BENEFICIARY TYPE': beneficiary_type,
            'SECTOR_code': sector_code,
            'Awc_code_new': awc_code
        }
        child_data.append(time_point_data)
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Process the input and make predictions
if submitted:
    # Convert input data to DataFrame
    input_df = pd.DataFrame(child_data)
    
    # Step 1: Encode categorical variables
    # Encode GENDER
    input_df['GENDER'] = input_df['GENDER'].map({'M': 1, 'F': 0})
    
    # Encode BENEFICIARY TYPE (one-hot encoding)
    input_df = pd.get_dummies(input_df, columns=['BENEFICIARY TYPE'], prefix='BENEFICIARY')
    
    # Ensure all beneficiary columns exist
    for col in ['BENEFICIARY_children_0m_6m', 'BENEFICIARY_children_6m_3y', 'BENEFICIARY_children_3y_6y']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Encode SECTOR_code and Awc_code_new for each target
    input_df['SECTOR_code_encoded_underweight'] = input_df['SECTOR_code'].map(underweight_sector_encoding).fillna(underweight_sector_encoding.mean())
    input_df['Awc_code_new_encoded_underweight'] = input_df['Awc_code_new'].map(underweight_awc_encoding).fillna(underweight_awc_encoding.mean())
    input_df['SECTOR_code_encoded_stunted'] = input_df['SECTOR_code'].map(stunted_sector_encoding).fillna(stunted_sector_encoding.mean())
    input_df['Awc_code_new_encoded_stunted'] = input_df['Awc_code_new'].map(stunted_awc_encoding).fillna(stunted_awc_encoding.mean())
    input_df['SECTOR_code_encoded_wasted'] = input_df['SECTOR_code'].map(wasted_sector_encoding).fillna(wasted_sector_encoding.mean())
    input_df['Awc_code_new_encoded_wasted'] = input_df['Awc_code_new'].map(wasted_awc_encoding).fillna(wasted_awc_encoding.mean())
    
    # Step 2: Standardize numerical features
    numerical_cols = ['WEIGHT', 'HEIGHT', 'Age_in_month']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Step 3: Prepare sequences for each target
    sequence_length = 3
    
    # For UNDERWEIGHT
    underweight_sequence = input_df[feature_cols_underweight].values  # Shape: (3, 9)
    underweight_sequence = np.expand_dims(underweight_sequence, axis=0).astype(np.float32)  # Shape: (1, 3, 9)
    
    # Debug: Display preprocessed UNDERWEIGHT sequence as a DataFrame
    underweight_sequence_2d = underweight_sequence.reshape(sequence_length, len(feature_cols_underweight))
    underweight_sequence_df = pd.DataFrame(underweight_sequence_2d, columns=feature_cols_underweight)
    underweight_sequence_df.index = [f"Time Point {i+1}" for i in range(sequence_length)]
    
    # For STUNTED
    stunted_sequence = input_df[feature_cols_stunted].values  # Shape: (3, 9)
    stunted_sequence = np.expand_dims(stunted_sequence, axis=0).astype(np.float32)  # Shape: (1, 3, 9)
    
    # Debug: Display preprocessed STUNTED sequence as a DataFrame
    stunted_sequence_2d = stunted_sequence.reshape(sequence_length, len(feature_cols_stunted))
    stunted_sequence_df = pd.DataFrame(stunted_sequence_2d, columns=feature_cols_stunted)
    stunted_sequence_df.index = [f"Time Point {i+1}" for i in range(sequence_length)]
    
    # For WASTED
    wasted_sequence = input_df[feature_cols_wasted].values  # Shape: (3, 9)
    wasted_sequence = np.expand_dims(wasted_sequence, axis=0).astype(np.float32)  # Shape: (1, 3, 9)
    
    # Debug: Display preprocessed WASTED sequence as a DataFrame
    wasted_sequence_2d = wasted_sequence.reshape(sequence_length, len(feature_cols_wasted))
    wasted_sequence_df = pd.DataFrame(wasted_sequence_2d, columns=feature_cols_wasted)
    wasted_sequence_df.index = [f"Time Point {i+1}" for i in range(sequence_length)]
    
    # Step 4: Make predictions
    underweight_pred = underweight_model.predict(underweight_sequence, verbose=0)
    underweight_pred_class = np.argmax(underweight_pred, axis=1)[0]
    underweight_label = ['normal', 'moderately underweight', 'severely underweight'][underweight_pred_class]
    
    stunted_pred = stunted_model.predict(stunted_sequence, verbose=0)
    stunted_pred_class = np.argmax(stunted_pred, axis=1)[0]
    stunted_label = ['normal', 'moderately stunted', 'severely stunted'][stunted_pred_class]
    
    wasted_pred = wasted_model.predict(wasted_sequence, verbose=0)
    wasted_pred_class = np.argmax(wasted_pred, axis=1)[0]
    wasted_label = ['normal', 'mam', 'sam'][wasted_pred_class]
    
    # Step 5: Display predictions
    st.write("### Predictions")
    st.write(f"**UNDERWEIGHT Status**: {underweight_label}")
    st.write(f"**STUNTED Status**: {stunted_label}")
    st.write(f"**WASTED Status**: {wasted_label}")