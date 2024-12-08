import joblib
import streamlit as st
from utils import get_feature_names
import pandas as pd

def load_models():
    """Load the saved model, scaler, and label encoder"""
    try:
        model_data = joblib.load('model/best_chronic_disease_model.joblib')
        scaler = joblib.load('model/feature_scaler.joblib')
        label_encoder = joblib.load('model/label_encoder.joblib')
        
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_names = model_data.get('feature_names', get_feature_names())
        else:
            model = model_data
            feature_names = get_feature_names()
            
        return model, scaler, label_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def prepare_features(data, feature_names):
    """Prepare features for model prediction"""
    features = {
        'MEM_GENDER_ENCODED': 1 if data['gender'] == "Male" else 0,
        'MEM_RACE_ENCODED': {"White": 0, "Black": 1, "Asian": 2, "Other": 3}[data['race']],
        'MEM_ETHNICITY_ENCODED': 1 if data['ethnicity'] == "Hispanic" else 0,
        'PAYER_LOB_ENCODED': 0,
        'SERVICE_SETTING_ENCODED': 0,
        'DIAGNOSTIC_CONDITION_CATEGORY_DESC_ENCODED': 0,
        'MEM_AGE_NUMERIC': data['age'],
        
        # Disease flags
        'HAS_HYPERTENSION': 1 if 'Hypertension' in data['conditions'] else 0,
        'HAS_DIABETES': 1 if 'Diabetes' in data['conditions'] else 0,
        'HAS_RENAL_FAILURE': 1 if 'Renal Failure' in data['conditions'] else 0,
        'HAS_OTHER_CHRONIC': 0,
        'HAS_CANCER': 1 if 'Cancer' in data['conditions'] else 0,
        'HAS_MENTAL_HEALTH': 1 if 'Depression' in data['conditions'] else 0,
        'HAS_HEART_FAILURE': 1 if 'Heart Failure' in data['conditions'] else 0,
        'HAS_ASTHMA': 1 if 'Asthma' in data['conditions'] else 0,
        'HAS_MUSCULOSKELETAL': 1 if 'Musculoskeletal' in data['conditions'] else 0,
        'HAS_NEUROLOGIC': 1 if 'Neurologic' in data['conditions'] else 0,
        'HAS_LIVER_DISEASE': 1 if 'Liver Disease' in data['conditions'] else 0,
        'HAS_DEMENTIA': 1 if 'Dementia' in data['conditions'] else 0,
        'HAS_CAD': 1 if 'CAD' in data['conditions'] else 0
    }
    
    # Add condition combinations
    features.update({
        'HAS_HYPERTENSION_AND_DIABETES': features['HAS_HYPERTENSION'] * features['HAS_DIABETES'],
        'HAS_HYPERTENSION_AND_HEART_FAILURE': features['HAS_HYPERTENSION'] * features['HAS_HEART_FAILURE'],
        'HAS_DIABETES_AND_RENAL_FAILURE': features['HAS_DIABETES'] * features['HAS_RENAL_FAILURE'],
        'HAS_HEART_FAILURE_AND_RENAL_FAILURE': features['HAS_HEART_FAILURE'] * features['HAS_RENAL_FAILURE'],
        'HAS_MENTAL_HEALTH_AND_NEUROLOGIC': features['HAS_MENTAL_HEALTH'] * features['HAS_NEUROLOGIC'],
        'HAS_LIVER_DISEASE_AND_RENAL_FAILURE': features['HAS_LIVER_DISEASE'] * features['HAS_RENAL_FAILURE']
    })
    
    # Calculate weighted risk score
    features['WEIGHTED_RISK_SCORE'] = sum([
        3 * (features['HAS_HEART_FAILURE'] + features['HAS_RENAL_FAILURE'] + features['HAS_CANCER']),
        2 * (features['HAS_LIVER_DISEASE'] + features['HAS_HYPERTENSION'] + features['HAS_DIABETES'] + 
             features['HAS_DEMENTIA'] + features['HAS_CAD']),
        1 * (features['HAS_MENTAL_HEALTH'] + features['HAS_ASTHMA'] + features['HAS_MUSCULOSKELETAL'] + 
             features['HAS_NEUROLOGIC'] + features['HAS_OTHER_CHRONIC'])
    ])
    
    return pd.DataFrame([features])[feature_names]