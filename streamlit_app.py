import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

def load_models():
    """Load the saved models and encoders"""
    try:
        model = joblib.load('models/chronic_disease_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        label_encoder = joblib.load('models/label_encoder.joblib')
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def predict_risk(patient_data, model, scaler, label_encoder):
    """Make predictions using the loaded model"""
    try:
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Scale features
        patient_scaled = scaler.transform(patient_df)
        
        # Get prediction and probabilities
        prediction = model.predict(patient_scaled)[0]
        probabilities = model.predict_proba(patient_scaled)[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_probs = probabilities[top_3_indices]
        top_3_labels = label_encoder.inverse_transform(top_3_indices)
        
        return {
            'predicted_condition': label_encoder.inverse_transform([prediction])[0],
            'confidence': f"{max(probabilities)*100:.1f}%",
            'risk_level': 'High' if max(probabilities) > 0.7 else 
                         'Medium' if max(probabilities) > 0.4 else 'Low',
            'top_3_predictions': [
                {'condition': str(label), 
                 'probability': f"{prob*100:.1f}%"}
                for label, prob in zip(top_3_labels, top_3_probs)
            ]
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    st.set_page_config(
        page_title="Chronic Disease Risk Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st.title("🏥 Chronic Disease Risk Predictor")
    st.markdown("""
    This tool predicts the risk of chronic diseases based on patient demographics and medical history.
    Please fill in the patient information below to get a risk assessment.
    """)
    
    # Load models
    model, scaler, label_encoder = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])
        
    with col2:
        st.subheader("Medical History")
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        cancer = st.checkbox("Cancer")
        renal_failure = st.checkbox("Renal Failure")
        depression = st.checkbox("Depression")
        heart_failure = st.checkbox("Heart Failure")
    
    # Create age group
    def get_age_group(age):
        if age < 18:
            return "Child"
        elif age < 35:
            return "Young Adult"
        elif age < 50:
            return "Adult"
        elif age < 65:
            return "Middle Age"
        else:
            return "Senior"
    
    if st.button("Predict Risk"):
        # Prepare input data
        age_group = get_age_group(age)
        input_data = {
            'MEM_AGE_NUMERIC': age,
            'MEM_GENDER_ENCODED': 1 if gender == "Female" else 0,
            'MEM_RACE_ENCODED': {"White": 0, "Black": 1, "Asian": 2, "Other": 3}[race],
            'MEM_ETHNICITY_ENCODED': 1 if ethnicity == "Hispanic" else 0,
            'PAYER_LOB_ENCODED': 0,  # Default value
            'SERVICE_SETTING_ENCODED': 1,  # Default value
            'HAS_HYPERTENSION': 1 if hypertension else 0,
            'HAS_DIABETES': 1 if diabetes else 0,
            'HAS_CANCER': 1 if cancer else 0,
            'HAS_RENAL FAILURE': 1 if renal_failure else 0,
            'HAS_DEPRESSION': 1 if depression else 0,
            'HAS_HEART FAILURE': 1 if heart_failure else 0,
            'AGE_GROUP_Adult': 1 if age_group == "Adult" else 0,
            'AGE_GROUP_Child': 1 if age_group == "Child" else 0,
            'AGE_GROUP_Middle Age': 1 if age_group == "Middle Age" else 0,
            'AGE_GROUP_Senior': 1 if age_group == "Senior" else 0,
            'AGE_GROUP_Unknown': 0,
            'AGE_GROUP_Young Adult': 1 if age_group == "Young Adult" else 0
        }
        
        # Make prediction
        result = predict_risk(input_data, model, scaler, label_encoder)
        
        # Display results
        st.markdown("---")
        st.subheader("Risk Assessment Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Primary Condition", result['predicted_condition'])
            
        with col2:
            st.metric("Confidence", result['confidence'])
            
        with col3:
            st.metric("Risk Level", result['risk_level'])
        
        # Display top 3 predictions
        st.subheader("Alternative Predictions")
        for pred in result['top_3_predictions']:
            st.write(f"Condition {pred['condition']}: {pred['probability']} probability")
        
        # Recommendations based on risk level
        st.subheader("Recommendations")
        if result['risk_level'] == 'High':
            st.warning("⚠️ Immediate medical consultation recommended")
        elif result['risk_level'] == 'Medium':
            st.info("ℹ️ Regular check-ups and lifestyle modifications recommended")
        else:
            st.success("✅ Continue maintaining healthy lifestyle habits")

if __name__ == "__main__":
    main()