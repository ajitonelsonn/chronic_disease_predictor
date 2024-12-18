import streamlit as st
import time
from styles import styles
from utils import load_condition_mapping, get_condition_name
from model_utils import load_models, prepare_features
from components import navigation, footer
from recommend import get_llm_recommendation
from database import save_prediction_to_database
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS additions
additional_styles = """
.stProgress > div > div > div > div {
    background-color: #6772e5;
}
.prediction-phase {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    background-color: #f7fafc;
    border-left: 4px solid #6772e5;
}
.metric-card:hover {
    transform: translateY(-2px);
    transition: all 0.2s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.custom-info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #ebf8ff;
    border: 1px solid #4299e1;
    margin: 1rem 0;
}
.progress-label {
    color: #4a5568;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
}
"""

def display_prediction_results(results, risk_level, patient_data):
    """Display prediction results in a modern format with patient information"""
    # Patient Information Summary
    st.markdown("""
        <div class='prediction-box'>
            <h3 style='color: #4a5568;'>📋 Patient Summary</h3>
            <p style='color: #718096;'>Demographics and Clinical Information</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display patient demographics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Age</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{patient_data['age']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Gender</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{patient_data['gender']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Race</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{patient_data['race']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Ethnicity</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{patient_data['ethnicity']}</p>
            </div>
        """, unsafe_allow_html=True)

    # Display selected conditions
    st.markdown("""
        <div class='prediction-box'>
            <h4 style='color: #4a5568;'>Selected Clinical Conditions</h4>
        </div>
    """, unsafe_allow_html=True)
    
    if patient_data['conditions']:
        for condition in patient_data['conditions']:
            st.markdown(f"""
                <div style='background-color: white; padding: 0.75rem; border-radius: 8px; margin: 0.3rem 0;
                          box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>
                    <span style='color: #4a5568;'>• {condition}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No pre-existing conditions selected")

    # Prediction Results with Progress Bar
    with st.spinner("Generating prediction visualization..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        progress_bar.empty()

    st.markdown("""
        <div class='prediction-box'>
            <h3 style='color: #4a5568;'>🔍 Prediction Results</h3>
            <p style='color: #718096;'>Analysis based on provided patient data</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Primary Condition</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{results['primary_condition']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Confidence</h4>
                <p style='font-size: 1.5rem; color: #2d3748;'>{results['confidence']*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_color = {
            "High": "#f56565",
            "Medium": "#ed8936",
            "Low": "#48bb78"
        }.get(risk_level, "#4a5568")
        
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #4a5568;'>Risk Level</h4>
                <p style='font-size: 1.5rem; color: {risk_color};'>{risk_level}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed predictions with progress animation
    st.markdown("""
        <div class='prediction-box'>
            <h4 style='color: #4a5568;'>Top 3 Most Likely Conditions</h4>
        </div>
    """, unsafe_allow_html=True)
    
    for i, (condition, prob) in enumerate(results['top_3_conditions']):
        # Progress animation for each condition
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        for j in range(int(prob * 100)):
            progress_bar.progress(j + 1)
            time.sleep(0.01)
        progress_placeholder.empty()
        
        st.markdown(f"""
            <div style='background-color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #4a5568; font-weight: 500;'>{condition}</span>
                    <span style='color: #718096;'>{prob*100:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def predict_conditions(features, model, scaler, label_encoder):
    """Make predictions using the model with progress tracking"""
    try:
        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load condition mapping
        status_text.text("Loading condition mappings...")
        progress_bar.progress(20)
        time.sleep(0.5)
        condition_map = load_condition_mapping()
        
        # Scale features
        status_text.text("Scaling features...")
        progress_bar.progress(40)
        time.sleep(0.5)
        features_scaled = scaler.transform(features)
        
        # Get prediction and probabilities
        status_text.text("Generating predictions...")
        progress_bar.progress(60)
        time.sleep(0.5)
        #prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Process results
        status_text.text("Processing results...")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_probs = probabilities[top_3_indices]
        top_3_conditions = label_encoder.inverse_transform(top_3_indices)
        
        # Map condition codes to descriptions
        primary_condition = get_condition_name(top_3_conditions[0], condition_map)
        top_3_mapped = [(get_condition_name(cond, condition_map), prob)
                       for cond, prob in zip(top_3_conditions, top_3_probs)]
        
        progress_bar.progress(100)
        status_text.text("Prediction complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return {
            'primary_condition': primary_condition,
            'confidence': top_3_probs[0],
            'top_3_conditions': top_3_mapped,
            'error': None
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'error': str(e)}

def process_and_save_results(results, patient_data, risk_level):
    """Process results and save to database with progress tracking"""
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        # Analysis phase - 30%
        st.markdown("<div class='progress-label'>Generating recommendations...</div>", unsafe_allow_html=True)
        analysis_progress = st.progress(0)
        for i in range(100):
            analysis_progress.progress(i + 1)
            time.sleep(0.01)
        
        # Recommendation phase - 30%
        #st.markdown("<div class='progress-label'>Generating recommendations...</div>", unsafe_allow_html=True)
        ##recommendation_progress = st.progress(0)
        condition_data = {
            'primary_condition': results['primary_condition'],
            'confidence': results['confidence'],
            'risk_level': risk_level,
            'condition_list': '\n'.join([f"- {cond}: {prob*100:.1f}%" 
                                       for cond, prob in results['top_3_conditions']])
        }
        recommendations = get_llm_recommendation(condition_data)
        #for i in range(30):
        #    recommendation_progress.progress(i + 1)
        #    time.sleep(0.01)
        
        # Database save phase - 40%
        st.markdown("<div class='progress-label'>Saving to database...</div>", unsafe_allow_html=True)
        save_progress = st.progress(0)
        save_success = save_prediction_to_database(
            patient_data=patient_data,
            prediction_results=results,
            recommendations=recommendations
        )
        for i in range(60):
            save_progress.progress(i + 1)
            time.sleep(0.01)
        
    progress_placeholder.empty()
    return recommendations, save_success

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Chronic Disease Risk Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    
    # Apply custom CSS
    st.markdown(f"<style>{styles}{additional_styles}</style>", unsafe_allow_html=True)
    
    # Navigation
    navigation()
    
    # Header
    st.markdown("""
    <div class="custom-header">
        <h1 style='text-align: center; color: white;'>Chronic Disease Risk Predictor</h1>
        <p style='text-align: center; color: #e2e8f0;'>
            AI-powered chronic disease risk assessment and personalized medical recommendations - Lōkahi Innovation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    model, scaler, label_encoder, feature_names = load_models()
    if model is None:
        return
    
    # Show New Prediction button if currently showing prediction
    if st.session_state.show_prediction:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.show_prediction = False
            st.rerun()

    # Show form only if not showing prediction
    if not st.session_state.show_prediction:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
                <div class='prediction-box'>
                    <h3 style='color: #4a5568;'>Patient Demographics</h3>
                    <p style='color: #718096;'>Enter basic patient information</p>
                </div>
            """, unsafe_allow_html=True)
            
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])

        with col2:
            st.markdown("""
                <div class='prediction-box'>
                    <h3 style='color: #4a5568;'>Clinical Conditions</h3>
                    <p style='color: #718096;'>Select all diagnosed conditions</p>
                </div>
            """, unsafe_allow_html=True)
            
            conditions = st.multiselect(
                "Select existing conditions",
                [
                    "Hypertension", "Diabetes", "Cancer", "Renal Failure",
                    "Depression", "Heart Failure", "Asthma", "CAD",
                    "Musculoskeletal", "Neurologic", "Liver Disease", "Dementia"
                ]
            )

        # Predict button
        if st.button("Generate Prediction", key="predict_button"):
            st.session_state.input_data = {
                'age': age,
                'gender': gender,
                'race': race,
                'ethnicity': ethnicity,
                'conditions': conditions
            }
            st.session_state.show_prediction = True
            st.rerun()
    
    # Show prediction results if state is true
    if st.session_state.show_prediction and hasattr(st.session_state, 'input_data'):
        with st.spinner("📊 Analyzing patient data..."):
            features = prepare_features(st.session_state.input_data, feature_names)
            results = predict_conditions(features, model, scaler, label_encoder)
            
            if results.get('error'):
                st.error(f"Error making prediction: {results['error']}")
                return
            
            risk_level = "High" if len(st.session_state.input_data['conditions']) >= 3 else \
                        "Medium" if len(st.session_state.input_data['conditions']) >= 1 else "Low"
            
            display_prediction_results(results, risk_level, st.session_state.input_data)
            
            try:
                with st.spinner("Processing and saving results..."):
                    recommendations, save_success = process_and_save_results(
                        results,
                        st.session_state.input_data,
                        risk_level
                    )
                    
                    if recommendations:
                        st.markdown("""
                            <div class='prediction-box'>
                                <h4 style='color: #4a5568;'>🎯 Personalized Recommendations</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if risk_level == "High":
                            st.warning(recommendations)
                        elif risk_level == "Medium":
                            st.info(recommendations)
                        else:
                            st.success(recommendations)
                        
                        if save_success:
                            st.snow()      
                    else:
                        st.error("Unable to generate recommendations. Please try again.")
            except Exception as e:
                logger.error(f"Processing error: {e}")
                st.error(f"Error processing results: {str(e)}")
    
    # Footer
    footer()

if __name__ == "__main__":
    main()