import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Framingham Heart Disease Dashboard",
    # page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the Framingham dataset"""
    url = "https://raw.githubusercontent.com/realaryagupta/Cardiovascular-Disease-Risk-Prediction/refs/heads/main/dataset/framingham.csv"
    try:
        data = pd.read_csv(url)
        return data
    except:
        st.error("Could not load data from URL. Please check your internet connection.")
        return None

def preprocess_data(data, handle_missing='drop'):
    """Preprocess the data"""
    data_processed = data.copy()
    
    if handle_missing == 'drop':
        data_processed = data_processed.dropna()
    elif handle_missing == 'impute':
        # Numerical columns
        numerical_cols = data_processed.select_dtypes(include=[np.number]).columns
        imputer_num = SimpleImputer(strategy='median')
        data_processed[numerical_cols] = imputer_num.fit_transform(data_processed[numerical_cols])
        
        # Categorical columns
        categorical_cols = data_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            data_processed[categorical_cols] = imputer_cat.fit_transform(data_processed[categorical_cols])
    
    return data_processed

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    fig = plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.1)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_age_distribution_plot(data):
    """Create age distribution plot by CHD status"""
    fig = px.violin(data, x='TenYearCHD', y='age', 
                    title='Age Distribution by CHD Status',
                    labels={'TenYearCHD': 'CHD Risk', 'age': 'Age'},
                    color='TenYearCHD',
                    color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
    fig.update_layout(showlegend=False)
    return fig

def create_risk_factors_plot(data):
    """Create risk factors comparison plot"""
    risk_factors = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=risk_factors,
        specs=[[{"secondary_y": False}]*4]*2
    )
    
    row_col_pairs = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3)]
    
    for i, factor in enumerate(risk_factors):
        if factor in data.columns:
            row, col = row_col_pairs[i]
            
            # Group by CHD status and calculate means
            chd_means = data.groupby('TenYearCHD')[factor].mean()
            
            fig.add_bar(
                x=['No CHD', 'CHD'],
                y=[chd_means[0], chd_means[1]],
                name=factor,
                row=row, col=col,
                showlegend=False,
                marker_color=['#1f77b4', '#ff7f0e']
            )
    
    fig.update_layout(height=600, title_text="Risk Factors Comparison by CHD Status")
    return fig

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models"""
    models = {}
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'predictions': lr_pred
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    return models, results

def main():
    st.markdown('<h1 class="main-header">❤️ Framingham Heart Disease Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Overview", "Data Exploration", "Visualizations", "Model Training", "Risk Prediction"])
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Missing data handling option
    st.sidebar.subheader("Data Processing Options")
    missing_data_option = st.sidebar.selectbox(
        "Handle Missing Data:",
        ["drop", "impute"],
        format_func=lambda x: "Drop rows with missing values" if x == "drop" else "Impute missing values"
    )
    
    # Process data
    processed_data = preprocess_data(data, missing_data_option)
    
    if page == "Overview":
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(processed_data))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", len(processed_data.columns) - 1)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            chd_cases = processed_data['TenYearCHD'].sum()
            st.metric("CHD Cases", chd_cases)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            chd_rate = (chd_cases / len(processed_data) * 100)
            st.metric("CHD Rate", f"{chd_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset info
        st.subheader("Dataset Information")
        st.write("**Shape:**", processed_data.shape)
        st.write("**Columns:**", list(processed_data.columns))
        
        # Missing values info
        if missing_data_option == "drop":
            missing_before = data.isnull().sum().sum()
            st.write(f"**Missing values removed:** {missing_before}")
            st.write(f"**Records after cleaning:** {len(processed_data)}")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(processed_data.head())
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(processed_data.describe())
    
    elif page == "Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        # Feature selection for exploration
        st.subheader("Feature Analysis")
        selected_feature = st.selectbox("Select a feature to analyze:", 
                                      processed_data.columns[:-1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig, ax = plt.subplots(figsize=(8, 6))
            processed_data[selected_feature].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title(f'Distribution of {selected_feature}')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            # Box plot by CHD status
            fig, ax = plt.subplots(figsize=(8, 6))
            processed_data.boxplot(column=selected_feature, by='TenYearCHD', ax=ax)
            ax.set_title(f'{selected_feature} by CHD Status')
            plt.suptitle('')  # Remove default title
            st.pyplot(fig)
        
        # Feature statistics
        st.subheader(f"Statistics for {selected_feature}")
        feature_stats = processed_data.groupby('TenYearCHD')[selected_feature].describe()
        st.dataframe(feature_stats)
        
        # Missing values analysis (if any)
        if data[selected_feature].isnull().sum() > 0:
            st.subheader("Missing Values Analysis")
            missing_pct = (data[selected_feature].isnull().sum() / len(data)) * 100
            st.write(f"Missing values in {selected_feature}: {missing_pct:.2f}%")
    
    elif page == "Visualizations":
        st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr_fig = create_correlation_heatmap(processed_data)
        st.pyplot(corr_fig)
        
        # Age distribution
        st.subheader("Age Distribution by CHD Status")
        age_fig = create_age_distribution_plot(processed_data)
        st.plotly_chart(age_fig, use_container_width=True)
        
        # Risk factors comparison
        st.subheader("Risk Factors Analysis")
        risk_fig = create_risk_factors_plot(processed_data)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Gender and smoking analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender Distribution")
            gender_chd = processed_data.groupby(['male', 'TenYearCHD']).size().unstack()
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_chd.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
            ax.set_title('CHD Cases by Gender')
            ax.set_xlabel('Gender (0=Female, 1=Male)')
            ax.set_ylabel('Count')
            ax.legend(['No CHD', 'CHD'])
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            if 'currentSmoker' in processed_data.columns:
                st.subheader("Smoking Status")
                smoking_chd = processed_data.groupby(['currentSmoker', 'TenYearCHD']).size().unstack()
                fig, ax = plt.subplots(figsize=(8, 6))
                smoking_chd.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                ax.set_title('CHD Cases by Smoking Status')
                ax.set_xlabel('Current Smoker (0=No, 1=Yes)')
                ax.set_ylabel('Count')
                ax.legend(['No CHD', 'CHD'])
                plt.xticks(rotation=0)
                st.pyplot(fig)
    
    elif page == "Model Training":
        st.markdown('<h2 class="sub-header">Machine Learning Model Training</h2>', unsafe_allow_html=True)
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            handle_imbalance = st.selectbox("Handle Class Imbalance", 
                                          ["None", "Oversample", "Undersample"])
        
        with col2:
            scale_features = st.checkbox("Scale Features", value=True)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        if st.button("Train Models"):
            # Prepare data
            X = processed_data.drop('TenYearCHD', axis=1)
            y = processed_data['TenYearCHD']
            
            # Handle class imbalance
            if handle_imbalance == "Oversample":
                oversample = RandomOverSampler(random_state=random_state)
                X, y = oversample.fit_resample(X, y)
                st.info("Applied oversampling to balance classes")
            elif handle_imbalance == "Undersample":
                undersample = RandomUnderSampler(random_state=random_state)
                X, y = undersample.fit_resample(X, y)
                st.info("Applied undersampling to balance classes")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                st.info("Features have been scaled")
            
            # Train models
            models, results = train_models(X_train, X_test, y_train, y_test)
            
            # Display results
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            for i, (model_name, metrics) in enumerate(results.items()):
                with col1 if i == 0 else col2:
                    st.write(f"**{model_name}**")
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, metrics['predictions'])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'{model_name} - Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
            
            # Feature importance (for Random Forest)
            if 'Random Forest' in models:
                st.subheader("Feature Importance (Random Forest)")
                feature_names = processed_data.drop('TenYearCHD', axis=1).columns
                importances = models['Random Forest'].feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                indices = np.argsort(importances)[::-1]
                
                ax.bar(range(len(importances)), importances[indices])
                ax.set_title('Feature Importance')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    elif page == "Risk Prediction":
        st.markdown('<h2 class="sub-header">CHD Risk Prediction</h2>', unsafe_allow_html=True)
        
        st.write("Enter patient information to predict CHD risk:")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=20, max_value=100, value=50)
                male = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                current_smoker = st.selectbox("Current Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col2:
                sys_bp = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
                dia_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
                tot_chol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200)
            
            with col3:
                bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
                heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150, value=70)
                glucose = st.number_input("Glucose", min_value=50, max_value=400, value=100)
            
            submitted = st.form_submit_button("Predict CHD Risk")
            
            if submitted:
                # Create a simple prediction model
                X = processed_data.drop('TenYearCHD', axis=1)
                y = processed_data['TenYearCHD']
                
                # Train a simple model for prediction
                model = LogisticRegression(random_state=42, max_iter=1000)
                
                # Scale the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                
                # Prepare input data
                input_data = np.array([[age, male, current_smoker, 0, 0, sys_bp, dia_bp, 
                                      tot_chol, bmi, heart_rate, glucose, 0]])
                
                # Ensure input has the same number of features as training data
                if input_data.shape[1] != X.shape[1]:
                    # Pad with zeros or adjust as needed
                    diff = X.shape[1] - input_data.shape[1]
                    if diff > 0:
                        padding = np.zeros((1, diff))
                        input_data = np.hstack([input_data, padding])
                    else:
                        input_data = input_data[:, :X.shape[1]]
                
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    color = "red" if prediction == 1 else "green"
                    st.markdown(f"**CHD Risk:** <span style='color: {color}'>{risk_level}</span>", 
                              unsafe_allow_html=True)
                
                with col2:
                    st.metric("Risk Probability", f"{probability:.2%}")
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "CHD Risk Percentage"},
                    delta = {'reference': 15},
                    gauge = {'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps' : [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}],
                            'threshold' : {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 90}}))
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()