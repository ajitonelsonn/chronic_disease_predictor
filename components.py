import streamlit as st
from datetime import datetime

def navigation():
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("streamlit_app.py")
            
    with col2:
        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard.py")

def footer():
    """Create footer section"""
    st.markdown("""
        <div class="footer">
            <h3>Chronic Disease Risk Predictor</h3>
            <p>Powered by Advanced AI and Machine Learning | Created by Ajito Nelson Lucio da Costa</p>
            <p style='font-size: 0.8rem;'>© {} Chronic Disease Risk Predictor. All rights reserved.</p>
        </div>
    """.format(datetime.now().year), unsafe_allow_html=True)