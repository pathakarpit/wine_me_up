import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()
# --- CONFIGURATION ---
st.set_page_config(page_title="Wine Quality AI", page_icon="🍷", layout="wide")

ARTIFACTS_PATH = os.path.join("app", "data", "eda_artifacts.joblib")
MODEL_ARTIFACTS_PATH = os.path.join("app", "data", "ema_artifacts.joblib")
API_URL = "http://127.0.0.1:8000/predict"

# ==========================================
# 1. DATA LOADING HELPER
# ==========================================
@st.cache_data
def load_artifacts():
    data = {}
    
    # Load EDA
    if os.path.exists(ARTIFACTS_PATH):
        try:
            data['eda'] = joblib.load(ARTIFACTS_PATH)
        except Exception as e:
            st.error(f"❌ Error loading EDA file: {e}")
    else:
        st.error(f"❌ Error: EDA file missing at {ARTIFACTS_PATH}")
        
    # Load Model Analysis
    if os.path.exists(MODEL_ARTIFACTS_PATH):
        try:
            data['ema'] = joblib.load(MODEL_ARTIFACTS_PATH)
        except Exception as e:
            st.warning(f"⚠️ Error loading Model Analysis: {e}")
    else:
        st.warning(f"⚠️ Model Analysis file missing at {MODEL_ARTIFACTS_PATH}")
        
    return data

# ==========================================
# 2. INDIVIDUAL PLOT RENDERERS
# ==========================================

def render_dataset_overview(data):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", data['shape'][0])
        st.metric("Total Features", data['shape'][1])
    with col2:
        st.metric("Duplicates Removed", data['duplicates_removed'])
        st.metric("Missing Values", data['missing'].sum())
    
    st.subheader("Data Preview")
    st.dataframe(data['head'], use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(data['description'], use_container_width=True)

def render_target_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=data['counts'].index, y=data['counts'].values, palette='viridis', ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def render_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(data['corr_matrix'], dtype=bool))
    sns.heatmap(data['corr_matrix'], mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

def render_boxplot_grid(data, full_data):
    if full_data is None: return
    features = data['features']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, col in enumerate(features):
        row, col_idx = divmod(i, 2)
        sns.boxplot(x='quality', y=col, data=full_data, hue='quality', palette='coolwarm', legend=False, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f"{col} vs Quality")
    st.pyplot(fig)

def render_kde_grid(data, full_data):
    if full_data is None: return
    features = data['features'][:6] # Limit to 6 for clean UI
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(data=full_data, x=col, hue='quality', palette='viridis', 
                   common_norm=False, fill=True, alpha=0.3, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
    plt.tight_layout()
    st.pyplot(fig)

def render_stratification_plot(data):
    dist_df = data['distribution']
    
    # FIX: Robustly handle index renaming
    # 1. Reset index to move the index (Quality IDs) into a column
    dist_reset = dist_df.reset_index()
    
    # 2. Force rename the first column to 'Quality Class' so we know exactly what to target
    # (This prevents errors if the column is named 'quality', 'index', or something else)
    dist_reset = dist_reset.rename(columns={dist_reset.columns[0]: "Quality Class"})
    
    # 3. Now we can safely melt using our known column name
    dist_melted = dist_reset.melt(id_vars='Quality Class', var_name='Set', value_name='Percentage')
    dist_melted['Quality Class'] = dist_melted['Quality Class'].astype(str)
    
    fig = px.bar(
        dist_melted, x='Quality Class', y='Percentage', color='Set', 
        barmode='group', title="Class Distribution: Train vs Test Splits", text_auto='.1%'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_baseline_leaderboard(data):
    results = data['results']
    st.markdown("#### 🏁 Baseline Results (Defaults)")
    st.dataframe(results, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x='Kappa Score', y='Model', palette='viridis', ax=ax)
    ax.set_title("Baseline Performance (Kappa)")
    ax.set_xlim(0, 0.7)
    st.pyplot(fig)

def render_tuned_leaderboard(data):
    results = data['results']
    st.markdown("#### 🏆 Final Leaderboard")
    st.dataframe(results.style.highlight_max(axis=0, subset=['Best Kappa Score']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x='Best Kappa Score', y='Model', palette='magma', ax=ax)
    ax.set_title("Hyperparameter Tuned Performance")
    ax.set_xlim(0, 0.7)
    st.pyplot(fig)
    
    with st.expander("🔎 View Best Hyperparameters"):
        for index, row in results.iterrows():
            st.markdown(f"**{row['Model']}**")
            st.json(row['Best Params'])
            st.divider()

def render_generic_plot(step, full_data):
    """Fallback for simple plot types"""
    data = step['data']
    st_type = step['type']

    if st_type == 'correlation_comparison':
        corrs = data['new_correlations']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=corrs.values, y=corrs.index, palette='coolwarm', ax=ax)
        st.pyplot(fig)
    elif st_type == 'outlier_plot':
        outliers = data['outliers']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=outliers.index, y=outliers.values, hue=outliers.index, palette='Reds_r', legend=False, ax=ax)
        st.pyplot(fig)
    elif st_type == 'profile_heatmap':
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(data['means_normalized'], annot=True, cmap='Blues', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    elif st_type == 'plotly':
        st.plotly_chart(data['fig'], use_container_width=True)
    elif st_type == 'feature_importance_plot':
        importances = data['importances']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', palette='viridis', legend=False, ax=ax)
        st.pyplot(fig)

# --- DISPATCHER ---
def dispatch_plot_rendering(step, full_data):
    """Routes the step data to the correct plotting function."""
    st_type = step['type']
    data = step['data']

    # EDA Types
    if st_type == 'dataset_overview': render_dataset_overview(data)
    elif st_type == 'plot': render_target_distribution(data)
    elif st_type == 'heatmap': render_heatmap(data)
    elif st_type == 'boxplot_grid': render_boxplot_grid(data, full_data)
    elif st_type == 'kde_grid': render_kde_grid(data, full_data)
    elif st_type == 'stratification_plot': render_stratification_plot(data)
    
    # Model Analysis Types
    elif st_type == 'model_leaderboard': # <--- ADDED THIS
        render_baseline_leaderboard(data)
    elif st_type == 'tuned_leaderboard': 
        render_tuned_leaderboard(data)
        
    # Generic Fallback
    else: render_generic_plot(step, full_data)

# ==========================================
# 3. PAGE RENDERERS
# ==========================================

def render_eda_page(artifacts):
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Deep dive into the chemical properties of wine.")
    
    if 'eda' not in artifacts:
        st.error("No EDA artifacts found. Please run 'perform_eda.py'.")
        return

    full_data = artifacts['eda'].get('full_data')
    sections = artifacts['eda'].get('eda_sections', [])
    
    for step in sections:
        st.markdown("---")
        st.header(step['title'])
        st.success(f"💡 **Insight:** \n\n{step['comment']}")
        
        # Call the dispatcher
        dispatch_plot_rendering(step, full_data)
        
        with st.expander(f"Show Python Code"):
            st.code(step['code'], language='python')

def render_models_page(artifacts):
    st.title("🧪 Model Selection & Tuning")
    st.markdown("Rigorous scientific process to select the best model.")

    if 'ema' not in artifacts:
        st.error("Model artifacts not found. Please run 'perform_model_analysis.py'.")
        return

    sections = artifacts['ema'].get('model_sections', [])
    for step in sections:
        st.markdown("---")
        st.header(step['title'])
        st.info(f"📝 **Methodology:** \n\n{step['comment']}")
        
        dispatch_plot_rendering(step, None) # Models page usually doesn't need raw DF
        
        with st.expander(f"Show Experiment Code"):
            st.code(step['code'], language='python')

def render_prediction_page():
    st.title("🤖 Live Wine Quality Predictor")
    st.write("Adjust the chemical properties below to predict wine quality.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # ... (Inputs for Alcohol, pH, etc. remain the same) ...
        with col1:
            alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)
            volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, 0.5)
            sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6)
            citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
        
        with col2:
            total_sulfur = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, 40.0)
            free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 70.0, 15.0)
            chlorides = st.number_input("Chlorides", 0.01, 0.6, 0.08)
            residual_sugar = st.number_input("Residual Sugar", 0.9, 15.0, 2.5)
        
        with col3:
            density = st.number_input("Density", 0.990, 1.005, 0.996, format="%.4f")
            pH = st.number_input("pH", 2.7, 4.0, 3.3)
            fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.0)
            
            # --- UPDATED MODEL SELECTOR ---
            # Map technical keys to friendly names
            model_display_names = {
                "default": "🏆 Best Model (Default)",
                "lightgbm_regressor": "🚀 LightGBM (Tuned - Secret Weapon)",
                "xgboost": "⚡ XGBoost",
                "catboost": "🐱 CatBoost",
                "random_forest": "🌲 Random Forest",
                "lightgbm_classifier": "💡 LightGBM (Classifier)",
                "svc": "📉 Support Vector Machine"
            }
            
            model_choice_key = st.selectbox(
                "Select Model Strategy", 
                options=list(model_display_names.keys()),
                format_func=lambda x: model_display_names.get(x)
            )

        submitted = st.form_submit_button("Predict Quality")
    
    if submitted:
        # Pass the 'model_choice_key' (technical name) to the handler
        handle_prediction_submission(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                     chlorides, free_sulfur, total_sulfur, density, pH, sulphates, 
                                     alcohol, model_choice_key)

def handle_prediction_submission(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                 chlorides, free_sulfur, total_sulfur, density, pH, sulphates, 
                                 alcohol, model_choice):
    payload = {
        "fixed_acidity": fixed_acidity, "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid, "residual_sugar": residual_sugar,
        "chlorides": chlorides, "free_sulfur_dioxide": free_sulfur,
        "total_sulfur_dioxide": total_sulfur, "density": density,
        "pH": pH, "sulphates": sulphates, "alcohol": alcohol
    }
    
    # --- FIX START: Define Headers with API Key ---
    # For local testing, we use the default key "my-secret-key"
    # In production, this comes from your environment variables
    api_key = os.getenv("API_KEY", "my-secret-key")
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    # --- FIX END ---
    
    try:
        # Pass headers=headers to the request
        response = requests.post(f"{API_URL}/{model_choice}", json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            quality = result['predicted_quality']
            
            st.markdown("### 🎯 Prediction Result")
            if quality >= 7:
                st.balloons()
                st.success(f"**Premium Quality Wine! (Score: {quality})**")
            elif quality >= 5:
                st.warning(f"**Average Table Wine (Score: {quality})**")
            else:
                st.error(f"**Poor Quality Wine (Score: {quality})**")
            
            with st.expander("See Raw JSON Response"):
                st.json(result)
        else:
            st.error(f"API Error: {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to backend. Is 'main.py' running? Error: {e}")

# ==========================================
# 4. MAIN CONTROLLER
# ==========================================

def render_home_page():
    st.title("🍷 WineMeUp: Intelligent Quality Prediction")
    st.markdown("### An End-to-End MLOps Project")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        **WineMeUp** is a full-stack AI application designed to analyze chemical properties 
        of wine and predict its quality with high precision.
        
        This project demonstrates a complete Data Science lifecycle:
        * **Data Engineering:** Automated pipelines for cleaning and validation.
        * **EDA:** Deep dive analysis into chemical correlations.
        * **Model Analysis:** A "Tournament" of 6 algorithms (XGBoost, CatBoost, etc.) 
            tuned via Bayesian Optimization (Optuna).
        * **Deployment:** FastAPI backend with Redis caching and Docker containerization.
        """)
        
        st.markdown("### 🛠️ Tech Stack")
        st.code("Python | Pandas | Scikit-Learn | Optuna | FastAPI | Redis | Streamlit | Docker | Nginx", language="text")

    with col2:
        st.info("🔗 **Project Resources**")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/pathakarpit/wine_me_up)")
        st.markdown("[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)")

    st.divider()
    st.success("👈 **Navigate using the sidebar to see the EDA, Model Experiments, or Live Prediction.**")

def main():
    st.sidebar.image("https://img.icons8.com/color/96/wine-bottle.png", width=80)
    st.sidebar.title("Navigation")
    
    # ADD "Project Overview" as the first option
    page = st.sidebar.radio("Go to:", [
        "🏠 Project Overview", 
        "1. EDA & Insights", 
        "2. Model Experiments", 
        "3. Live Prediction"
    ])

    artifacts = load_artifacts()

    if page == "🏠 Project Overview":
        render_home_page()
    elif page == "1. EDA & Insights":
        render_eda_page(artifacts)
    elif page == "2. Model Experiments":
        render_models_page(artifacts)
    elif page == "3. Live Prediction":
        render_prediction_page()

if __name__ == "__main__":
    main()