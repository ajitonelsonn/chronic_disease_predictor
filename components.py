import streamlit as st
from datetime import datetime

def navigation():
    """Create navigation menu"""
    st.markdown("""
        <div class="nav-menu">
            <a href="#" class="nav-link">🏠 Home</a>
            <a href="#about" class="nav-link">ℹ️ About</a>
            <a href="#contact" class="nav-link">📞 Contact</a>
            <a href="#help" class="nav-link">❓ Help</a>
        </div>
    """, unsafe_allow_html=True)

def footer():
    """Create footer section"""
    st.markdown("""
        <div class="footer">
            <h3>Chronic Disease Risk Predictor</h3>
            <p>Powered by Advanced AI and Machine Learning | Created by Ajito Nelson Lucio da Costa</p>
            <p style='font-size: 0.8rem;'>© {} Chronic Disease Risk Predictor. All rights reserved.</p>
        </div>
    """.format(datetime.now().year), unsafe_allow_html=True)
