import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

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
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
        <div class="custom-header">
            <h1 style='text-align: center; color: #2c3e50;'>🏥 Chronic Disease Risk Predictor</h1>
            <p style='text-align: center; color: #7f8c8d;'>
                Advanced AI-powered tool for predicting chronic disease risks based on patient demographics and medical history
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load models
    model, scaler, label_encoder = load_models()
    
    if model is None:
        st.error("⚠️ System initialization failed. Please contact support.")
        return

    # Create two columns for the form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="prediction-box">
                <h3 style='color: #2c3e50;'>📋 Patient Demographics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
        ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])

    with col2:
        st.markdown("""
            <div class="prediction-box">
                <h3 style='color: #2c3e50;'>🏥 Medical History</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            hypertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            cancer = st.checkbox("Cancer")
            
        with col2_2:
            renal_failure = st.checkbox("Renal Failure")
            depression = st.checkbox("Depression")
            heart_failure = st.checkbox("Heart Failure")

    # Predict button with styling
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Risk"):
        # Prepare input data
        age_group = get_age_group(age)
        input_data = prepare_input_data(age, gender, race, ethnicity, 
                                      hypertension, diabetes, cancer, 
                                      renal_failure, depression, heart_failure,
                                      age_group)
        
        # Make prediction
        result = predict_risk(input_data, model, scaler, label_encoder)
        
        # Display results in a modern layout
        st.markdown("""
            <div class="prediction-box">
                <h2 style='color: #2c3e50;'>🎯 Risk Assessment Results</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for metrics
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #7f8c8d;'>Primary Condition</h4>
                    <h2 style='color: #2c3e50;'>{}</h2>
                </div>
            """.format(result['predicted_condition']), unsafe_allow_html=True)
            
        with m2:
            st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #7f8c8d;'>Confidence</h4>
                    <h2 style='color: #2c3e50;'>{}</h2>
                </div>
            """.format(result['confidence']), unsafe_allow_html=True)
            
        with m3:
            st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #7f8c8d;'>Risk Level</h4>
                    <h2 style='color: {};'>{}</h2>
                </div>
            """.format(
                "#e74c3c" if result['risk_level'] == "High" else 
                "#f39c12" if result['risk_level'] == "Medium" else "#27ae60",
                result['risk_level']
            ), unsafe_allow_html=True)

        # Alternative Predictions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="prediction-box">
                <h3 style='color: #2c3e50;'>🔄 Alternative Predictions</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for pred in result['top_3_predictions']:
            st.markdown(f"""
                <div style='background-color: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;'>
                    <p style='margin: 0;'><b>{pred['condition']}</b>: {pred['probability']}</p>
                </div>
            """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="prediction-box">
                <h3 style='color: #2c3e50;'>💡 Recommendations</h3>
            </div>
        """, unsafe_allow_html=True)

        if result['risk_level'] == 'High':
            st.warning("⚠️ Immediate medical consultation recommended")
            recommendations = [
                "Schedule an appointment with your healthcare provider",
                "Monitor symptoms closely",
                "Review and maintain medication compliance if applicable"
            ]
        elif result['risk_level'] == 'Medium':
            st.info("ℹ️ Regular monitoring recommended")
            recommendations = [
                "Schedule routine check-ups",
                "Maintain healthy lifestyle habits",
                "Consider preventive measures"
            ]
        else:
            st.success("✅ Continue maintaining healthy habits")
            recommendations = [
                "Regular exercise",
                "Balanced diet",
                "Regular check-ups"
            ]

        for rec in recommendations:
            st.markdown(f"""
                <div style='background-color: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;'>
                    <p style='margin: 0;'>• {rec}</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()