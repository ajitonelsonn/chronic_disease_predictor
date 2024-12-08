<div align="center">
  <img src="https://raw.githubusercontent.com/ajitonelsonn/chronic_disease_predictor/main/Images/1.png" alt="LAFAEK AI" width="400"/>
</div>

# Chronic Disease Risk Predictor 🏥

**An advanced AI-powered tool for predicting chronic disease risks and providing personalized medical recommendations.**

---

## 🌟 Features

- **Risk Prediction**: Utilizes XGBoost model to predict chronic disease risks
- **Interactive UI**: Modern, user-friendly interface built with Streamlit
- **Real-time Analysis**: Instant risk assessment and recommendations
- **AI Recommendations**: Personalized health recommendations using LLM (Together AI)
- **Comprehensive Reports**: Detailed analysis of patient risk factors

---

## 🔧 Technology Stack

- **Frontend**: Streamlit (v1.40.2)
- **Backend**: Python 3.9+
- **ML Framework**: XGBoost (v2.1.3)
- **Data Processing**: Pandas (v2.2.3), NumPy (v2.1.3)
- **Model Serialization**: Joblib (v1.4.2)
- **Recommendations**: Together AI Integration

---

## 🚀 Getting Started

### Prerequisites

Ensure you have:

- Python 3.8+
- pip

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ajitonelsonn/chronic_disease_predictor.git
   cd chronic_disease_predictor
   ```

2. **Set up a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:  
   Create `.streamlit/secrets.toml` and add your Together AI API key:

   ```toml
   [api_keys]
   togetherapi = "your_api_key_here"
   ```

---

## 📊 Model Performance

```plaintext
Model Performance Metrics:
- Accuracy: 81.76%
- Average Precision: 57.59%
- Average Recall: 59.66%
- Average F1-Score: 33762.05
```

---

## 📈 Workflow

```mermaid
graph TD
    A[Patient Data Input] --> B[Data Preprocessing]
    B --> C[Risk Assessment]
    C --> D[XGBoost Model]
    D --> E[Risk Prediction]
    E --> F[LLM Analysis]
    F --> G[Recommendations]

    subgraph "Data Processing"
    B
    end

    subgraph "ML Pipeline"
    C
    D
    E
    end

    subgraph "AI Recommendations"
    F
    G
    end
```

---

## 📁 Project Structure

```plaintext
chronic_disease_predictor/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── create_model/
│   └── Chronic_Disease_Risk_Prediction_Model.ipynb
├── data/
│   └── PRIMARY_CHRONIC_CONDITION_ROLLUP_DESC.csv
├── model/
│   ├── best_chronic_disease_model.joblib
│   ├── feature_scaler.joblib
│   └── label_encoder.joblib
├── components.py
├── model_utils.py
├── recommend.py
├── streamlit_app.py
├── styles.py
├── utils.py
└── requirements.txt
```

---

## 🛠 Development

1. **Start the Streamlit app**:

   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the app in your browser**:

   Open [http://localhost:8501](http://localhost:8501).

---

## 👥 Author

**Ajito Nelson Lucio da Costa**

[![Facebook](https://img.shields.io/badge/Facebook-%40ajitonelsonn-blue)](https://facebook.com/kharu.kharu89/)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%40ajitonelson-blue)](https://linkedin.com/in/ajitonelson)

---

<div align="center">

**Built with ❤️ in Timor-Leste 🇹🇱**

</div>
