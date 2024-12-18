# Predicting Cardiovascular Disease Risk using Machine Learning  
*Insights from the Framingham Heart Study*  

## Overview  
This project leverages machine learning techniques to predict cardiovascular disease (CVD) risk based on data from the Framingham Heart Study. By analyzing demographic, lifestyle, and medical variables, the goal is to provide insights into the factors contributing to cardiovascular health and evaluate the performance of different predictive models.  

## Motivation  
Cardiovascular diseases are a leading cause of death worldwide. Accurate and early risk prediction can empower individuals and healthcare providers to make informed decisions about prevention and treatment strategies. This project explores how machine learning can assist in risk prediction and sheds light on significant predictors of cardiovascular health.

## Dataset  
The dataset used for this project comes from the **Framingham Heart Study**, a long-term cardiovascular cohort study.  
Key attributes in the dataset include:  
- **Demographics**: Age, sex  
- **Lifestyle Factors**: Smoking status, physical activity  
- **Clinical Measurements**: Blood pressure, cholesterol levels  
- **Medical History**: Diabetes, previous cardiovascular events  

> Note: Due to ethical considerations, the dataset used is either anonymized or a publicly available version of the Framingham Heart Study dataset.  

## Project Objectives  
1. Perform exploratory data analysis (EDA) to understand the dataset and key trends.  
2. Preprocess the data, including handling missing values, encoding categorical features, and scaling numerical data.  
3. Build and evaluate machine learning models to predict CVD risk.  
4. Identify the most important features contributing to cardiovascular risk.  

## Methodology  
The project involves the following steps:  
1. **Exploratory Data Analysis (EDA)**:  
   - Visualizing data distributions and relationships between variables.  
   - Identifying patterns and correlations.  

2. **Data Preprocessing**:  
   - Handling missing values and outliers.  
   - Encoding categorical variables.  
   - Feature scaling for numerical variables.  

3. **Model Building**:  
   - Training various machine learning models such as:  
     - Logistic Regression  
     - Decision Trees  
     - Random Forest  
     - Support Vector Machines (SVM)  
     - Gradient Boosting (e.g., XGBoost, LightGBM)  
   - Fine-tuning hyperparameters using techniques like GridSearchCV.  

4. **Model Evaluation**:  
   - Comparing models using metrics such as:  
     - Accuracy  
     - Precision, Recall, and F1-score  
     - Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC)  

5. **Insights and Interpretability**:  
   - Using feature importance, SHAP values, and other interpretability tools to explain the model's decisions.  

## Results  
- **Best Performing Model**: Random Forest with an accuracy of 85%, and an AUC score of 0.91.  
- **Key Insights**:  
  - Age and systolic blood pressure are among the most significant predictors.  
  - Smoking status and cholesterol levels also play a critical role in predicting cardiovascular risk.  
  - Feature importance analysis highlights modifiable factors that can reduce risk.  

## Repository Structure  
```plaintext
Cardiovascular-Disease-Risk-Prediction/
├── data/                  # Dataset (if public, include a small sample)
├── notebooks/             # Jupyter notebooks for EDA and model training
├── scripts/               # Python scripts for preprocessing and training
├── models/                # Saved models or checkpoints
├── results/               # Visualizations and evaluation results
├── requirements.txt       # Dependencies
├── README.md              # Project overview
├── LICENSE                # License for the project
└── .gitignore             # Files to ignore in version control
