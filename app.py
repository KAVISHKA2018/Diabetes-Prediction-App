import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Set page configuration
st.set_page_config(
    page_title="DiabetesGuard: AI-Powered Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86AB;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86AB;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 0.4rem;
        font-size: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)


# Load the model and scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/diabetes.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #2E86AB;'>DiabetesGuard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 2rem;'>🩺</div>", unsafe_allow_html=True)

nav_options = {
    "Home": "🏠",
    "Data Exploration": "🔍",
    "Visualizations": "📊",
    "Model Prediction": "🤖",
    "Model Performance": "📈"
}
page = st.sidebar.radio("Navigate to:", list(nav_options.keys()), 
                        format_func=lambda x: f"{nav_options[x]} {x}")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #e6f2ff; padding: 1rem; border-radius: 10px;'>
    <h4 style='color: #2E86AB; margin-top: 0;'>About This App</h4>
    <p style='font-size: 0.9rem; margin-bottom: 0;'>
    This app uses machine learning to predict diabetes risk based on health metrics.
    The model was trained on the Pima Indians Diabetes Dataset.
    </p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
    <p>Built with Streamlit</p>
    <p>Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)


# ------------------- PAGES -------------------
# Home Page
if page == "Home":
    st.image("dbp_background.PNG", width=1150)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>DiabetesGuard</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #6c757d;'>AI-Powered Diabetes Prediction</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #2E86AB; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin-top: 0;'>Early Detection Saves Lives</h2>
        <p style='font-size: 1.2rem;'>Predict diabetes risk with 78% accuracy using our machine learning model</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-value'>768</div><div class='metric-label'>Total Samples</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-value'>8</div><div class='metric-label'>Features</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-value'>268</div><div class='metric-label'>Diabetes Cases</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-value'>77.9%</div><div class='metric-label'>Model Accuracy</div></div>", unsafe_allow_html=True)

    st.markdown("<h3 class='sub-header'>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)


# ------------------- Model Prediction -------------------
elif page == "Model Prediction":
    st.markdown("<h1 class='main-header'>Diabetes Prediction</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #e6f2ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
        Enter the patient's health metrics below to predict the likelihood of diabetes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=100, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, step=1)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=846, value=80, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0, step=0.1, format="%.1f")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.5, step=0.01, format="%.3f")
        age = st.number_input("Age", min_value=21, max_value=81, value=30, step=1)

    # Input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown("""
        <div style='background-color: #ffcccc; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h2 style='color: #d63031; margin: 0;'>Diabetes Detected</h2>
            <p style='font-size: 1.2rem;'>Please consult a healthcare professional</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h2 style='color: #28a745; margin: 0;'>No Diabetes Detected</h2>
            <p style='font-size: 1.2rem;'>Maintain a healthy lifestyle</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability with Progress Bar
    diabetes_prob = probability[0][1]
    st.markdown(f"### Probability of Diabetes: **{diabetes_prob*100:.2f}%**")
    st.progress(diabetes_prob)
