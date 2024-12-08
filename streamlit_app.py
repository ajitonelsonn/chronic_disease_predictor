import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Chronic Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .custom-header {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #0366d6;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def load_condition_mapping():
    """Load the condition code to description mapping"""
    try:
        mapping_df = pd.read_csv('data/PRIMARY_CHRONIC_CONDITION_ROLLUP_DESC.csv')
        return dict(zip(mapping_df['PCC_CODE'], mapping_df['PCC_ROLLUP']))
    except Exception as e:
        st.error(f"Error loading condition mapping: {str(e)}")
        return {}

def get_condition_name(code, condition_map):
    """Convert condition code to actual condition name"""
    return condition_map.get(float(code), f"Unknown ({code})")

def get_feature_names():
    """Get expected feature names in correct order"""
    return [
        'MEM_GENDER_ENCODED',
        'MEM_RACE_ENCODED',
        'MEM_ETHNICITY_ENCODED',
        'PAYER_LOB_ENCODED',
        'SERVICE_SETTING_ENCODED',
        'DIAGNOSTIC_CONDITION_CATEGORY_DESC_ENCODED',
        'HAS_HYPERTENSION',
        'HAS_DIABETES',
        'HAS_RENAL_FAILURE',
        'HAS_OTHER_CHRONIC',
        'HAS_CANCER',
        'HAS_MENTAL_HEALTH',
        'HAS_HEART_FAILURE',
        'HAS_ASTHMA',
        'HAS_MUSCULOSKELETAL',
        'HAS_NEUROLOGIC',
        'HAS_LIVER_DISEASE',
        'HAS_DEMENTIA',
        'HAS_CAD',
        'HAS_HYPERTENSION_AND_DIABETES',
        'HAS_HYPERTENSION_AND_HEART_FAILURE',
        'HAS_DIABETES_AND_RENAL_FAILURE',
        'HAS_HEART_FAILURE_AND_RENAL_FAILURE',
        'HAS_MENTAL_HEALTH_AND_NEUROLOGIC',
        'HAS_LIVER_DISEASE_AND_RENAL_FAILURE',
        'MEM_AGE_NUMERIC',
        'WEIGHTED_RISK_SCORE'
    ]

def load_models():
    """Load the saved model, scaler, and label encoder"""
    try:
        model_data = joblib.load('model/best_chronic_disease_model.joblib')
        scaler = joblib.load('model/feature_scaler.joblib')
        label_encoder = joblib.load('model/label_encoder.joblib')
        
        # Handle both new and old model formats
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
    # Create base features
    features = {
        'MEM_GENDER_ENCODED': 0 if data['gender'] == "Male" else 1,
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
    
    # Create DataFrame with correct feature order
    features_df = pd.DataFrame([features])
    
    # Ensure all required features exist and are in correct order
    for feature in feature_names:
        if feature not in features_df.columns:
            features_df[feature] = 0
            
    return features_df[feature_names]

def predict_conditions(features, model, scaler, label_encoder):
    """Make predictions using the model"""
    try:
        # Load condition mapping
        condition_map = load_condition_mapping()
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get prediction and probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_probs = probabilities[top_3_indices]
        top_3_conditions = label_encoder.inverse_transform(top_3_indices)
        
        # Map condition codes to descriptions
        primary_condition = get_condition_name(top_3_conditions[0], condition_map)
        top_3_mapped = [(get_condition_name(cond, condition_map), prob) 
                       for cond, prob in zip(top_3_conditions, top_3_probs)]
        
        return {
            'primary_condition': primary_condition,
            'confidence': top_3_probs[0],
            'top_3_conditions': top_3_mapped,
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    # Header
    st.markdown("""
        <div class="custom-header">
            <h1 style='text-align: center; color: #2c3e50;'>Chronic Disease Risk Predictor</h1>
            <p style='text-align: center; color: #7f8c8d;'>
                AI-powered disease risk prediction using advanced machine learning
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load models
    model, scaler, label_encoder, feature_names = load_models()
    if model is None:
        return

    # Create form layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='prediction-box'><h3>Patient Information</h3></div>", 
                   unsafe_allow_html=True)
        
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])

    with col2:
        st.markdown("<div class='prediction-box'><h3>Medical Conditions</h3></div>", 
                   unsafe_allow_html=True)
        
        conditions = st.multiselect(
            "Select existing conditions",
            [
                "Hypertension", "Diabetes", "Cancer", "Renal Failure",
                "Depression", "Heart Failure", "Asthma", "CAD",
                "Musculoskeletal", "Neurologic", "Liver Disease", "Dementia"
            ]
        )

    # Predict button
    if st.button("Generate Prediction"):
        with st.spinner("Analyzing patient data..."):
            # Prepare data
            input_data = {
                'age': age,
                'gender': gender,
                'race': race,
                'ethnicity': ethnicity,
                'conditions': conditions
            }
            
            # Generate features
            features = prepare_features(input_data, feature_names)
            
            # Make prediction
            results = predict_conditions(features, model, scaler, label_encoder)
            
            if results.get('error'):
                st.error(f"Error making prediction: {results['error']}")
                return
            
            # Display results
            st.markdown("<div class='prediction-box'><h3>Prediction Results</h3></div>", 
                       unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Primary Predicted Condition", results['primary_condition'])
            
            with col2:
                st.metric("Confidence", f"{results['confidence']*100:.1f}%")
            
            with col3:
                risk_level = "High" if len(conditions) >= 3 else "Medium" if len(conditions) >= 1 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Detailed predictions
            st.markdown("#### Top 3 Most Likely Conditions")
            for condition, prob in results['top_3_conditions']:
                st.markdown(f"- **{condition}**: {prob*100:.1f}%")
            
            # Risk assessment and recommendations
            st.markdown("#### Recommendations")
            if risk_level == "High":
                st.warning("""
                    - Schedule immediate consultation with healthcare provider
                    - Review current medications and treatment plans
                    - Monitor symptoms closely
                    - Consider lifestyle modifications as recommended by your doctor
                """)
            elif risk_level == "Medium":
                st.info("""
                    - Schedule regular check-ups
                    - Monitor your condition
                    - Maintain a healthy lifestyle
                    - Follow prescribed treatment plans
                """)
            else:
                st.success("""
                    - Continue regular health check-ups
                    - Maintain healthy lifestyle habits
                    - Stay active and maintain a balanced diet
                    - Monitor any changes in health
                """)

if __name__ == "__main__":
    main()