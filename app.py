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
    .nav-item {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .nav-item:hover {
        background-color: #e6f2ff;
    }
    .nav-item.active {
        background-color: #2E86AB;
        color: white;
    }
    .performance-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
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

# Create a custom navigation in sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #2E86AB;'>DiabetesGuard</h2>", unsafe_allow_html=True)

# Add logo or icon
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 2rem;'>🩺</div>", unsafe_allow_html=True)

# Navigation options with icons
nav_options = {
    "Home": "🏠",
    "Data Exploration": "🔍",
    "Visualizations": "📊",
    "Model Prediction": "🤖",
    "Model Performance": "📈"
}

# Create navigation
page = st.sidebar.radio("Navigate to:", list(nav_options.keys()), 
                        format_func=lambda x: f"{nav_options[x]} {x}")

# Add some spacing
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# Add information about the app
st.sidebar.markdown("""
<div style='background-color: #e6f2ff; padding: 1rem; border-radius: 10px;'>
    <h4 style='color: #2E86AB; margin-top: 0;'>About This App</h4>
    <p style='font-size: 0.9rem; margin-bottom: 0;'>
    This app uses machine learning to predict diabetes risk based on health metrics.
    The model was trained on the Pima Indians Diabetes Dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
    <p>Built with using Streamlit</p>
    <p>Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)

# Home page
if page == "Home":

    # Cover image
    st.image("dbp_background.PNG", width=1150)

    # Create header with banner
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>DiabetesGuard</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #6c757d;'>AI-Powered Diabetes Prediction</h3>", unsafe_allow_html=True)
    
    # banner image 
    # For now, using a colored banner with text
    st.markdown("""
    <div style='background-color: #2E86AB; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin-top: 0;'>Early Detection Saves Lives</h2>
        <p style='font-size: 1.2rem;'>Predict diabetes risk with 78% accuracy using our machine learning model</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p>This application predicts the likelihood of diabetes based on patient health metrics using a machine learning model trained on the Pima Indians Diabetes Dataset.</p>
        <p><strong>Navigate using the sidebar</strong> to explore the data, visualize patterns, make predictions, and evaluate model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display dataset overview with custom cards
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>768</div>
            <div class='metric-label'>Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>8</div>
            <div class='metric-label'>Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>268</div>
            <div class='metric-label'>Diabetes Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>77.9%</div>
            <div class='metric-label'>Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show sample data
    st.markdown("<h3 class='sub-header'>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

# Data Exploration page
elif page == "Data Exploration":
    st.markdown("<h1 class='main-header'>Data Exploration</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
    st.write(f"Shape: {df.shape}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>Data Types</h3>", unsafe_allow_html=True)
        st.write(df.dtypes)
    with col2:
        st.markdown("<h3>Missing Values</h3>", unsafe_allow_html=True)
        st.write(df.isnull().sum())
    
    st.markdown("<h2 class='sub-header'>Statistical Summary</h2>", unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("<h2 class='sub-header'>Filter Data</h2>", unsafe_allow_html=True)
    columns = st.multiselect("Select columns to display", df.columns, default=df.columns.tolist())
    filtered_df = df[columns]
    
    # Add filters
    st.write("Apply Filters:")
    col1, col2 = st.columns(2)
    with col1:
        min_glucose = st.slider("Minimum Glucose", int(df['Glucose'].min()), int(df['Glucose'].max()), int(df['Glucose'].min()))
    with col2:
        max_glucose = st.slider("Maximum Glucose", int(df['Glucose'].min()), int(df['Glucose'].max()), int(df['Glucose'].max()))
    
    filtered_df = filtered_df[(filtered_df['Glucose'] >= min_glucose) & (filtered_df['Glucose'] <= max_glucose)]
    
    st.dataframe(filtered_df, use_container_width=True)

# Visualizations page
elif page == "Visualizations":
    st.markdown("<h1 class='main-header'>Data Visualizations</h1>", unsafe_allow_html=True)
    
    chart_type = st.selectbox("Select Chart Type", 
                             ["Distribution Plots", "Correlation Heatmap", "Outcome Comparison", "Pair Plot", "Feature vs Outcome"])
    
    if chart_type == "Distribution Plots":
        st.markdown("<h2 class='sub-header'>Feature Distributions</h2>", unsafe_allow_html=True)
        feature = st.selectbox("Select Feature", df.columns[:-1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, ax=ax, color="#2E86AB")
        ax.set_title(f'Distribution of {feature}')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
    elif chart_type == "Correlation Heatmap":
        st.markdown("<h2 class='sub-header'>Correlation Matrix</h2>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        
    elif chart_type == "Outcome Comparison":
        st.markdown("<h2 class='sub-header'>Comparison by Diabetes Outcome</h2>", unsafe_allow_html=True)
        feature = st.selectbox("Select Feature", df.columns[:-1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Outcome', y=feature, data=df, ax=ax, palette=["#2E86AB", "#A23B72"])
        ax.set_title(f'{feature} by Diabetes Outcome')
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
    elif chart_type == "Pair Plot":
        st.markdown("<h2 class='sub-header'>Pair Plot of Features</h2>", unsafe_allow_html=True)
        st.info("This may take a moment to load...")
        
        features = st.multiselect("Select Features", df.columns[:-1], default=['Glucose', 'BMI', 'Age'])
        
        if features:
            fig = sns.pairplot(df[features + ['Outcome']], hue='Outcome', palette=["#2E86AB", "#A23B72"])
            st.pyplot(fig)
            
    elif chart_type == "Feature vs Outcome":
        st.markdown("<h2 class='sub-header'>Feature vs Outcome Scatter Plot</h2>", unsafe_allow_html=True)
        x_feature = st.selectbox("X-axis Feature", df.columns[:-1], index=0)
        y_feature = st.selectbox("Y-axis Feature", df.columns[:-1], index=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_feature, y=y_feature, hue='Outcome', data=df, ax=ax, palette=["#2E86AB", "#A23B72"])
        ax.set_title(f'{x_feature} vs {y_feature} by Outcome')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

# Model Prediction page
elif page == "Model Prediction":
    st.markdown("<h1 class='main-header'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #e6f2ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
        Enter the patient's health metrics below to predict the likelihood of diabetes.
        The model will provide a prediction and confidence score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        glucose = st.slider("Glucose Level", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    
    with col2:
        insulin = st.slider("Insulin", 0, 846, 80)
        bmi = st.slider("BMI", 0.0, 67.1, 25.0)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
        age = st.slider("Age", 21, 81, 30)
    
    with col3:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 100%;'>
            <h4 style='color: #2E86AB; margin-top: 0;'>Normal Ranges</h4>
            <ul style='padding-left: 1.2rem;'>
                <li>Glucose: 70-100 mg/dL (fasting)</li>
                <li>Blood Pressure: < 120/80 mmHg</li>
                <li>BMI: 18.5-24.9</li>
                <li>Skin Thickness: 10-40 mm</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Create feature array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    # Display results
    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction[0] == 1:
            st.markdown("""
            <div style='background-color: #ffcccc; padding: 2rem; border-radius: 10px; text-align: center;'>
                <h2 style='color: #d63031; margin: 0;'>Diabetes Detected</h2>
                <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Please consult a healthcare professional</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 2rem; border-radius: 10px; text-align: center;'>
                <h2 style='color: #28a745; margin: 0;'>No Diabetes Detected</h2>
                <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Maintain a healthy lifestyle</p>
            </div>
            """, unsafe_allow_html=True)
    
    with result_col2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px;'>
            <h3 style='color: #2E86AB; margin-top: 0;'>Probability Analysis</h3>
            <p style='font-size: 1.2rem;'><strong>Probability of Diabetes:</strong> {:.2f}%</p>
            <p style='font-size: 1.2rem;'><strong>Confidence:</strong> {:.2f}%</p>
        </div>
        """.format(probability[0][1]*100, max(probability[0])*100), unsafe_allow_html=True)
    
    # Progress bar for probability
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Diabetes Risk Level:**")
    st.progress(probability[0][1])
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
        feature_importance = pd.DataFrame({
            'Feature': df.columns[:-1],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax, color="#2E86AB")
        ax.set_title('Feature Importance')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

# Model Performance page
elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance Evaluation</h1>", unsafe_allow_html=True)
    
    # Model comparison section
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
    
    model_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'SVM'],
        'CV Score': ['76.88%', '78.18%', '76.88%'],
        'CV Std (±)': ['6.74%', '2.49%', '3.58%'],
        'Test Accuracy': ['77.92%', '70.78%', '74.03%']
    }
    
    model_df = pd.DataFrame(model_data)
    
    # Display as a styled table
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #2E86AB; margin-top: 0;'>Performance Metrics Comparison</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a styled dataframe
    st.dataframe(model_df.style.highlight_max(subset=['Test Accuracy'], color='#d4edda'), use_container_width=True)
    
    # Add visual comparison using matplotlib instead of plotly
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(model_data['Model']))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars
    ax.bar(r1, [76.88, 78.18, 76.88], width=bar_width, label='CV Score', color='#2E86AB')
    ax.bar(r2, [77.92, 70.78, 74.03], width=bar_width, label='Test Accuracy', color='#A23B72')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks([r + bar_width/2 for r in range(len(model_data['Model']))])
    ax.set_xticklabels(model_data['Model'])
    ax.legend()
    ax.set_facecolor('#f8f9fa')
    
    # Display the chart
    st.pyplot(fig)
    
    # Best model info
    st.markdown("""
    <div style='background-color: #e6f2ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #2E86AB; margin-top: 0;'>Best Performing Model</h3>
        <p style='font-size: 1.2rem;'><strong>Random Forest</strong> with <strong>77.92%</strong> test accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("<h2 class='sub-header'>Confusion Matrix</h2>", unsafe_allow_html=True)
    
    # confusion matrix data
    cm = np.array([[87, 13], [21, 33]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification Report
    st.markdown("<h2 class='sub-header'>Classification Report</h2>", unsafe_allow_html=True)
    
    # classification report
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px;'>
        <pre style='font-family: monospace; font-size: 1rem; margin: 0;'>
              precision    recall  f1-score   support

           0       0.81      0.87      0.84       100
           1       0.72      0.61      0.66        54

    accuracy                           0.78       154
   macro avg       0.76      0.74      0.75       154
weighted avg       0.77      0.78      0.77       154
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics breakdown
    st.markdown("<h2 class='sub-header'>Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>77.9%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>81.0%</div>
            <div class='metric-label'>Precision (Class 0)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>72.0%</div>
            <div class='metric-label'>Precision (Class 1)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>87.0%</div>
            <div class='metric-label'>Recall (Class 0)</div>
        </div>
        """, unsafe_allow_html=True)