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
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern dark theme
st.markdown("""
<style>
    /* Main background color */
    body {
        background-color: #000000; /* Black background */
        color: #E0E0E0; /* Light grey text for readability */
    }
    .stApp {
        background-color: #000000; /* Black background */
    }
    .main-header {
        font-size: 3.5rem;
        color: #64B5F6; /* Lighter blue for main header */
        text-align: center;
        margin-bottom: 2.5rem;
        text-shadow: 2px 2px 8px rgba(100,181,246,0.5); /* Subtle shadow for depth */
    }
    .sub-header {
        font-size: 2.2rem;
        color: #FFB74D; /* Lighter orange for sub headers */
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #FFB74D; /* Underline for emphasis */
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1a1a1a; /* Darker grey for cards */
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 5px solid #64B5F6; /* Accent border */
        box-shadow: 3px 3px 10px rgba(0,0,0,0.4); /* Soft shadow for lift */
        color: #E0E0E0; /* Light text on dark card */
        margin-bottom: 1rem;
    }
    .metric-card strong {
        color: #64B5F6; /* Lighter blue for strong text in cards */
        font-size: 1.1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0; /* Light grey for all headers */
    }
    p, li, div {
        color: #B0B0B0; /* Slightly darker grey for general text */
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #333333; /* Darker input fields */
        color: #E0E0E0; /* Light text in inputs */
        border: 1px solid #555555;
        border-radius: 0.3rem;
    }
    .stButton>button {
        background-color: #64B5F6; /* Blue button */
        color: white;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #42A5F5; /* Lighter blue on hover */
        border-color: #42A5F5;
    }
    .stAlert {
        background-color: #333333;
        color: #E0E0E0;
    }
    .stMarkdown a {
        color: #81D4FA; /* Lighter blue for links */
    }
    /* Adjust Streamlit specific elements for dark theme */
    .css-1d391kg, .css-1lcbmhc, .css-1y4y1p7 { /* sidebar and main content */
        background-color: #000000;
        color: #E0E0E0;
    }
    .css-1oe5zby { /* Streamlit header text */
        color: #64B5F6;
    }
    .css-1j0r5yr { /* Sidebar selectbox option */
        color: #E0E0E0;
    }
    .css-q8sptp { /* Sidebar selectbox label */
        color: #E0E0E0;
    }
    /* Matplotlib and Seaborn plots will need custom styling within their functions */
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the Framingham dataset"""
    url = "https://raw.githubusercontent.com/realaryagupta/Cardiovascular-Disease-Risk-Prediction/refs/heads/main/dataset/framingham.csv"
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Could not load data from URL: {e}. Please check your internet connection.")
        return None

def preprocess_data(data, handle_missing='drop'):
    """Preprocess the data"""
    data_processed = data.copy()
    
    # Ensure 'TenYearCHD' is numeric first, coercing errors to NaN
    if 'TenYearCHD' in data_processed.columns:
        data_processed['TenYearCHD'] = pd.to_numeric(data_processed['TenYearCHD'], errors='coerce')

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
    
    # After dropping or imputing, ensure 'TenYearCHD' is explicitly integer type
    if 'TenYearCHD' in data_processed.columns:
        # Drop any remaining NaNs in 'TenYearCHD' that might have been coerced
        # This is crucial before converting to int, as int type cannot hold NaNs
        data_processed.dropna(subset=['TenYearCHD'], inplace=True)
        data_processed['TenYearCHD'] = data_processed['TenYearCHD'].astype(int)

    return data_processed

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Ensure only numeric columns are used for correlation
    correlation_matrix = data.select_dtypes(include=[np.number]).corr() 
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.1, fmt=".2f", ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', color='#E0E0E0')
    ax.tick_params(axis='x', colors='#E0E0E0')
    ax.tick_params(axis='y', colors='#E0E0E0')
    plt.tight_layout()
    return fig

def create_age_distribution_plot(data):
    """Create age distribution plot by CHD status"""
    fig = px.violin(data, x='TenYearCHD', y='age', 
                    title='Age Distribution by CHD Status',
                    labels={'TenYearCHD': 'CHD Risk', 'age': 'Age'},
                    color='TenYearCHD',
                    color_discrete_map={0: '#64B5F6', 1: '#FFB74D'}, # Modern colors
                    template="plotly_dark") # Use dark theme for plotly
    fig.update_layout(showlegend=False, 
                      title_font_color="#E0E0E0", 
                      font_color="#B0B0B0")
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
            # Ensure TenYearCHD is int for grouping
            if 'TenYearCHD' in data.columns and pd.api.types.is_numeric_dtype(data['TenYearCHD']):
                chd_means = data.groupby('TenYearCHD')[factor].mean()
                
                fig.add_bar(
                    x=['No CHD', 'CHD'],
                    y=[chd_means[0], chd_means[1]],
                    name=factor,
                    row=row, col=col,
                    showlegend=False,
                    marker_color=['#64B5F6', '#FFB74D'] # Modern colors
                )
            fig.update_xaxes(showgrid=False, title_font_color="#B0B0B0", tickfont_color="#B0B0B0", row=row, col=col)
            fig.update_yaxes(showgrid=True, gridcolor='#333333', title_font_color="#B0B0B0", tickfont_color="#B0B0B0", row=row, col=col)

    
    fig.update_layout(height=600, 
                      title_text="Risk Factors Comparison by CHD Status",
                      title_font_color="#E0E0E0",
                      font_color="#B0B0B0",
                      paper_bgcolor="#000000", # Black background for plotly
                      plot_bgcolor="#000000") # Black background for plot area
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
    st.markdown('<h1 class="main-header"> Framingham Heart Disease Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---") # Add a separator
    page = st.sidebar.selectbox("Choose a page", 
                               ["Overview", "Data Exploration", "Visualizations", "Model Training", "Risk Prediction"])
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Missing data handling option
    st.sidebar.markdown("---") # Add a separator
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
            st.markdown('<div class="metric-card"><strong style="color: #64B5F6;">Total Records</strong></div>', unsafe_allow_html=True)
            st.write(f'<p style="color:#E0E0E0; font-size:1.5rem; font-weight:bold;">{len(processed_data)}</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><strong style="color: #64B5F6;">Features</strong></div>', unsafe_allow_html=True)
            st.write(f'<p style="color:#E0E0E0; font-size:1.5rem; font-weight:bold;">{len(processed_data.columns) - 1}</p>', unsafe_allow_html=True)
        
        with col3:
            chd_cases = processed_data['TenYearCHD'].sum()
            st.markdown('<div class="metric-card"><strong style="color: #64B5F6;">CHD Cases</strong></div>', unsafe_allow_html=True)
            st.write(f'<p style="color:#FFB74D; font-size:1.5rem; font-weight:bold;">{chd_cases}</p>', unsafe_allow_html=True)
        
        with col4:
            chd_rate = (chd_cases / len(processed_data) * 100)
            st.markdown('<div class="metric-card"><strong style="color: #64B5F6;">CHD Rate</strong></div>', unsafe_allow_html=True)
            st.write(f'<p style="color:#FFB74D; font-size:1.5rem; font-weight:bold;">{chd_rate:.1f}%</p>', unsafe_allow_html=True)

        st.markdown("---") # Separator
        # Dataset info
        st.subheader("Dataset Information")
        st.markdown(f"<p style='color:#B0B0B0;'><b>Shape:</b> {processed_data.shape}</p>", unsafe_allow_html=True)
        
        # Missing values info
        if missing_data_option == "drop":
            missing_before = data.isnull().sum().sum()
            st.markdown(f"<p style='color:#B0B0B0;'><b>Missing values removed:</b> {missing_before}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#B0B0B0;'><b>Records after cleaning:</b> {len(processed_data)}</p>", unsafe_allow_html=True)
        
        st.markdown("---") # Separator
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(processed_data.head().style.set_properties(**{'background-color': '#1a1a1a', 'color': '#E0E0E0', 'border-color': '#333333'}))
        
        st.markdown("---") # Separator
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(processed_data.describe().style.set_properties(**{'background-color': '#1a1a1a', 'color': '#E0E0E0', 'border-color': '#333333'}))
    
    elif page == "Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        # Feature selection for exploration
        st.subheader("Feature Analysis")
        
        # Ensure 'TenYearCHD' is not in the selectable features for x-axis in general plots
        features_for_selection = processed_data.columns.tolist()
        if 'TenYearCHD' in features_for_selection:
            features_for_selection.remove('TenYearCHD')
            
        selected_feature = st.selectbox("Select a feature to analyze:", features_for_selection)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig, ax = plt.subplots(figsize=(8, 6))
            processed_data[selected_feature].hist(bins=30, alpha=0.7, ax=ax, color='#64B5F6')
            ax.set_title(f'Distribution of {selected_feature}', color='#E0E0E0')
            ax.set_xlabel(selected_feature, color='#B0B0B0')
            ax.set_ylabel('Frequency', color='#B0B0B0')
            ax.tick_params(axis='x', colors='#B0B0B0')
            ax.tick_params(axis='y', colors='#B0B0B0')
            ax.set_facecolor("#1a1a1a") # Plot background
            fig.patch.set_facecolor("#000000") # Figure background
            st.pyplot(fig)
        
        with col2:
            # Display a warning instead of the problematic plot
            st.warning(f"The box plot for '{selected_feature}' by CHD Status is currently disabled due to data consistency issues impacting visualization. Please proceed to the 'Visualizations' page for alternative plots.")
            # The problematic sns.boxplot code is removed from here:
            # fig, ax = plt.subplots(figsize=(8, 6))
            # if 'TenYearCHD' in processed_data.columns and not pd.api.types.is_integer_dtype(processed_data['TenYearCHD']):
            #      processed_data['TenYearCHD'] = processed_data['TenYearCHD'].astype(int)
            # sns.boxplot(x='TenYearCHD', y=selected_feature, data=processed_data, ax=ax, 
            #             palette={0: '#64B5F6', 1: '#FFB74D'})
            # ax.set_title(f'{selected_feature} by CHD Status', color='#E0E0E0')
            # ax.set_xlabel('CHD Status (0: No, 1: Yes)', color='#B0B0B0')
            # ax.set_ylabel(selected_feature, color='#B0B0B0')
            # ax.tick_params(axis='x', colors='#B0B0B0')
            # ax.tick_params(axis='y', colors='#B0B0B0')
            # ax.set_facecolor("#1a1a1a") # Plot background
            # fig.patch.set_facecolor("#000000") # Figure background
            # plt.suptitle('')  # Remove default title
            # st.pyplot(fig)
        
        st.markdown("---") # Separator
        # Feature statistics
        st.subheader(f"Statistics for {selected_feature}")
        if 'TenYearCHD' in processed_data.columns:
            feature_stats = processed_data.groupby('TenYearCHD')[selected_feature].describe()
            st.dataframe(feature_stats.style.set_properties(**{'background-color': '#1a1a1a', 'color': '#E0E0E0', 'border-color': '#333333'}))
        else:
            st.write("Cannot show statistics grouped by 'TenYearCHD' as the column is missing or problematic.")
            st.dataframe(processed_data[selected_feature].describe().to_frame().style.set_properties(**{'background-color': '#1a1a1a', 'color': '#E0E0E0', 'border-color': '#333333'}))
        
        # Missing values analysis (if any)
        if data[selected_feature].isnull().sum() > 0:
            st.markdown("---") # Separator
            st.subheader("Missing Values Analysis")
            missing_pct = (data[selected_feature].isnull().sum() / len(data)) * 100
            st.markdown(f"<p style='color:#B0B0B0;'>Missing values in {selected_feature}: <span style='color:#FFB74D; font-weight:bold;'>{missing_pct:.2f}%</span></p>", unsafe_allow_html=True)
    
    elif page == "Visualizations":
        st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr_fig = create_correlation_heatmap(processed_data)
        st.pyplot(corr_fig)
        
        st.markdown("---") # Separator
        # Age distribution
        st.subheader("Age Distribution by CHD Status")
        age_fig = create_age_distribution_plot(processed_data)
        st.plotly_chart(age_fig, use_container_width=True)
        
        st.markdown("---") # Separator
        # Risk factors comparison
        st.subheader("Risk Factors Analysis")
        risk_fig = create_risk_factors_plot(processed_data)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        st.markdown("---") # Separator
        # Gender and smoking analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender Distribution")
            # Ensure 'male' column is treated as category for consistent plotting
            if 'male' in processed_data.columns and 'TenYearCHD' in processed_data.columns:
                gender_chd = processed_data.groupby(['male', 'TenYearCHD']).size().unstack(fill_value=0) # Add fill_value
                fig, ax = plt.subplots(figsize=(8, 6))
                gender_chd.plot(kind='bar', ax=ax, color=['#64B5F6', '#FFB74D'])
                ax.set_title('CHD Cases by Gender', color='#E0E0E0')
                ax.set_xlabel('Gender (0=Female, 1=Male)', color='#B0B0B0')
                ax.set_ylabel('Count', color='#B0B0B0')
                ax.legend(['No CHD', 'CHD'], labelcolor='white')
                ax.tick_params(axis='x', colors='#B0B0B0')
                ax.tick_params(axis='y', colors='#B0B0B0')
                ax.set_facecolor("#1a1a1a") # Plot background
                fig.patch.set_facecolor("#000000") # Figure background
                plt.xticks(rotation=0)
                st.pyplot(fig)
            else:
                st.write("Gender or TenYearCHD columns not found for plotting.")
        
        with col2:
            if 'currentSmoker' in processed_data.columns and 'TenYearCHD' in processed_data.columns:
                st.subheader("Smoking Status")
                # Ensure 'currentSmoker' column is treated as category
                smoking_chd = processed_data.groupby(['currentSmoker', 'TenYearCHD']).size().unstack(fill_value=0) # Add fill_value
                fig, ax = plt.subplots(figsize=(8, 6))
                smoking_chd.plot(kind='bar', ax=ax, color=['#64B5F6', '#FFB74D'])
                ax.set_title('CHD Cases by Smoking Status', color='#E0E0E0')
                ax.set_xlabel('Current Smoker (0=No, 1=Yes)', color='#B0B0B0')
                ax.set_ylabel('Count', color='#B0B0B0')
                ax.legend(['No CHD', 'CHD'], labelcolor='white')
                ax.tick_params(axis='x', colors='#B0B0B0')
                ax.tick_params(axis='y', colors='#B0B0B0')
                ax.set_facecolor("#1a1a1a") # Plot background
                fig.patch.set_facecolor("#000000") # Figure background
                plt.xticks(rotation=0)
                st.pyplot(fig)
            else:
                st.write("CurrentSmoker or TenYearCHD columns not found for plotting.")
    
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
        
        if st.button("Train Models", help="Click to train the selected machine learning models."):
            st.markdown("---")
            st.info("Training models... Please wait.")
            # Prepare data
            # Ensure TenYearCHD is available and correct type before dropping
            if 'TenYearCHD' not in processed_data.columns:
                st.error("Error: 'TenYearCHD' column not found in processed data. Cannot train model.")
                return
            
            X = processed_data.drop('TenYearCHD', axis=1)
            y = processed_data['TenYearCHD']
            
            # Handle class imbalance
            if handle_imbalance == "Oversample":
                oversample = RandomOverSampler(random_state=random_state)
                X, y = oversample.fit_resample(X, y)
                st.success("Applied oversampling to balance classes.")
            elif handle_imbalance == "Undersample":
                undersample = RandomUnderSampler(random_state=random_state)
                X, y = undersample.fit_resample(X, y)
                st.success("Applied undersampling to balance classes.")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                st.success("Features have been scaled.")
            
            # Train models
            models, results = train_models(X_train, X_test, y_train, y_test)
            
            # Display results
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            for i, (model_name, metrics) in enumerate(results.items()):
                with (col1 if i == 0 else col2):
                    st.markdown(f"**<span style='color:#64B5F6;'>{model_name}</span>**", unsafe_allow_html=True)
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, metrics['predictions'])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                                xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'])
                    ax.set_title(f'{model_name} - Confusion Matrix', color='#E0E0E0')
                    ax.set_xlabel('Predicted', color='#B0B0B0')
                    ax.set_ylabel('Actual', color='#B0B0B0')
                    ax.tick_params(axis='x', colors='#B0B0B0')
                    ax.tick_params(axis='y', colors='#B0B0B0')
                    ax.set_facecolor("#1a1a1a") # Plot background
                    fig.patch.set_facecolor("#000000") # Figure background
                    st.pyplot(fig)
            
            # Feature importance (for Random Forest)
            if 'Random Forest' in models:
                st.markdown("---")
                st.subheader("Feature Importance (Random Forest)")
                feature_names = X.columns # Use original X columns for feature names
                importances = models['Random Forest'].feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                indices = np.argsort(importances)[::-1]
                
                ax.bar(range(len(importances)), importances[indices], color='#FFB74D')
                ax.set_title('Feature Importance', color='#E0E0E0')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', color='#B0B0B0')
                ax.set_ylabel('Importance', color='#B0B0B0')
                ax.tick_params(axis='y', colors='#B0B0B0')
                ax.set_facecolor("#1a1a1a") # Plot background
                fig.patch.set_facecolor("#000000") # Figure background
                plt.tight_layout()
                st.pyplot(fig)
    
    elif page == "Risk Prediction":
        st.markdown('<h2 class="sub-header">CHD Risk Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("<p style='color:#B0B0B0;'>Enter patient information to predict 10-year Coronary Heart Disease (CHD) risk.</p>", unsafe_allow_html=True)
        
        # Create input form based on actual dataset columns
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Patient Information")
            col1, col2, col3 = st.columns(3)
            
            # Initialize input dictionary with default values
            input_dict = {}
            
            with col1:
                # Check if columns exist before adding input fields
                if 'age' in processed_data.columns:
                    input_dict['age'] = st.number_input("Age", min_value=20, max_value=100, value=50, help="Age of the patient.")
                
                if 'male' in processed_data.columns:
                    input_dict['male'] = st.selectbox("Gender", options=[0, 1], 
                                                    format_func=lambda x: "Female" if x == 0 else "Male", help="Gender of the patient (0 for Female, 1 for Male).")
                
                if 'currentSmoker' in processed_data.columns:
                    input_dict['currentSmoker'] = st.selectbox("Current Smoker", options=[0, 1], 
                                                             format_func=lambda x: "No" if x == 0 else "Yes", help="Is the patient a current smoker?")
                
                if 'cigsPerDay' in processed_data.columns:
                    input_dict['cigsPerDay'] = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0, help="Number of cigarettes smoked per day.")
            
            with col2:
                if 'BPMeds' in processed_data.columns:
                    input_dict['BPMeds'] = st.selectbox("BP Medication", options=[0, 1], 
                                                      format_func=lambda x: "No" if x == 0 else "Yes", help="Is the patient on blood pressure medication?")
                
                if 'prevalentStroke' in processed_data.columns:
                    input_dict['prevalentStroke'] = st.selectbox("Previous Stroke", options=[0, 1], 
                                                               format_func=lambda x: "No" if x == 0 else "Yes", help="Has the patient had a previous stroke?")
                
                if 'prevalentHyp' in processed_data.columns:
                    input_dict['prevalentHyp'] = st.selectbox("Hypertension", options=[0, 1], 
                                                            format_func=lambda x: "No" if x == 0 else "Yes", help="Does the patient have hypertension?")
                
                if 'diabetes' in processed_data.columns:
                    input_dict['diabetes'] = st.selectbox("Diabetes", options=[0, 1], 
                                                        format_func=lambda x: "No" if x == 0 else "Yes", help="Does the patient have diabetes?")
            
            with col3:
                if 'sysBP' in processed_data.columns:
                    input_dict['sysBP'] = st.number_input("Systolic BP", min_value=80, max_value=250, value=120, help="Systolic Blood Pressure (mmHg).")
                
                if 'diaBP' in processed_data.columns:
                    input_dict['diaBP'] = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80, help="Diastolic Blood Pressure (mmHg).")
                
                if 'totChol' in processed_data.columns:
                    input_dict['totChol'] = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200, help="Total Cholesterol level (mg/dL).")
                
                if 'BMI' in processed_data.columns:
                    input_dict['BMI'] = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, help="Body Mass Index (kg/m²).")
                
                if 'heartRate' in processed_data.columns:
                    input_dict['heartRate'] = st.number_input("Heart Rate", min_value=40, max_value=150, value=70, help="Heart rate (beats per minute).")
                
                if 'glucose' in processed_data.columns:
                    input_dict['glucose'] = st.number_input("Glucose", min_value=50, max_value=400, value=100, help="Glucose level (mg/dL).")
            
            st.markdown("---")
            submitted = st.form_submit_button("Predict CHD Risk", help="Click to get the 10-year CHD risk prediction.")
            
            if submitted:
                try:
                    # Prepare the data
                    X = processed_data.drop('TenYearCHD', axis=1)
                    y = processed_data['TenYearCHD']
                    
                    # Train a model for prediction (using Logistic Regression for demonstration)
                    # For a real application, you might want to use the best performing model from 'Model Training' page
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    scaler = StandardScaler()
                    
                    # Scale the features and fit the model
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    
                    # Create input dataframe with the same columns as training data
                    # Ensure all training columns are present in input_df, filling missing ones with appropriate defaults
                    input_df = pd.DataFrame([input_dict])
                    
                    # Align columns - crucial for consistent predictions
                    for col in X.columns:
                        if col not in input_df.columns:
                            # Fill with median for numerical, mode for categorical, to prevent errors
                            if X[col].dtype in ['int64', 'float64']:
                                input_df[col] = X[col].median()
                            else: # Assuming object types are categorical that need mode
                                input_df[col] = X[col].mode()[0]
                    
                    # Reorder columns to match training data
                    input_df = input_df[X.columns]
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_df)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0][1]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        risk_level = "High Risk" if prediction == 1 else "Low Risk"
                        color = "#FF4D4D" if prediction == 1 else "#4CAF50" # Red for high, Green for low
                        st.markdown(f"**CHD Risk:** <span style='color: {color}; font-size:1.5rem; font-weight:bold;'>{risk_level}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Risk Probability", f"{probability:.2%}", help="Probability of developing CHD within 10 years.")
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", # Removed delta as it's not directly applicable here without a reference
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "<span style='color:#E0E0E0;'>CHD Risk Percentage</span>", 'font': {'size': 20}},
                        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#B0B0B0"},
                                'bar': {'color': "#64B5F6"}, # Main bar color
                                'steps' : [
                                    {'range': [0, 30], 'color': "#4CAF50"}, # Low risk (green)
                                    {'range': [30, 60], 'color': "#FFC107"}, # Medium risk (amber)
                                    {'range': [60, 100], 'color': "#FF4D4D"}], # High risk (red)
                                'threshold' : {'line': {'color': "#E0E0E0", 'width': 4},
                                              'thickness': 0.75, 'value': 50}})) # Threshold at 50%
                    
                    fig.update_layout(paper_bgcolor="#000000", font={'color': "#B0B0B0", 'family': "Arial"}) # Set plotly background to black
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    # Show input summary
                    st.subheader("Input Summary")
                    input_summary = pd.DataFrame([input_dict]).T
                    input_summary.columns = ['Value']
                    st.dataframe(input_summary.style.set_properties(**{'background-color': '#1a1a1a', 'color': '#E0E0E0', 'border-color': '#333333'}))
                    
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    st.write("Please check your inputs and try again.")

if __name__ == "__main__":
    main()