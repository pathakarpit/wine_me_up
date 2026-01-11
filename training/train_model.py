import pandas as pd
import numpy as np
import joblib
import os
import sys

# Import our project settings
import train_utils

# Import Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def load_and_clean_data():
    """
    Loads data and performs EDA-based cleaning.
    """
    print(f"Loading data from {train_utils.DATA_FILE_PATH}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(train_utils.DATA_FILE_PATH)
    except FileNotFoundError:
        sys.exit(f"Error: Data file not found at {train_utils.DATA_FILE_PATH}")

    # 2. Drop irrelevant columns (Found in EDA)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    # 3. Ensure Quality is an Integer (Crucial for Classification)
    df['quality'] = df['quality'].astype(int)
    
    # 4. Handle Duplicates (Good practice for production)
    df = df.drop_duplicates() # Optional: decide if you want to keep them

    print(f"Data Loaded. Shape: {df.shape}")
    return df


def train_models(df):
    """
    Trains all candidate models on the full dataset.
    Returns a dictionary of trained model objects.
    """
    print("\n--- Starting Training Session ---")
    
    X = df.drop('quality', axis=1)
    y = df['quality']

    # 1. Define the Model Architectures
    # (These are just the blueprints, not trained yet)
    model_blueprints = {
        # --- Standard Baselines (Default Settings) ---
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        "lightgbm_classifier": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "catboost": CatBoostClassifier(verbose=0, random_state=42),
        "svc": SVC(kernel='rbf', C=1.0, random_state=42),
        
        # --- The Tuned "Secret Weapon" (Optuna Findings) ---
        # We inject the specific values that gave us the 0.566 Kappa Score
        "lightgbm_regressor": lgb.LGBMRegressor(
            random_state=42, 
            verbose=-1,
            learning_rate=0.0146,       # Found by Optuna (Slow learning)
            n_estimators=1129,          # Found by Optuna (High iterations)
            max_depth=3,                # Found by Optuna (Shallow trees)
            num_leaves=85,
            subsample=0.903,
            colsample_bytree=0.555,
            reg_alpha=0.433,
            reg_lambda=0.683
        )
    }

    trained_models = {}

    # 2. Train Loop
    for name, model in model_blueprints.items():
        print(f"Training {name}...")
        
        # Special Logic: XGBoost needs 0-based labels
        if name == "xgboost":
            y_train = y - 3
        else:
            y_train = y
            
        model.fit(X, y_train)
        trained_models[name] = model
        print(f"   -> Trained.")
        
    return trained_models



def save_models(trained_models):
    """
    Saves a dictionary of trained models to disk as .joblib files.
    """
    print("\n--- Saving Models ---")

    # Ensure the directory exists
    if not os.path.exists(train_utils.MODEL_DIR):
        os.makedirs(train_utils.MODEL_DIR)
        print(f"Created directory: {train_utils.MODEL_DIR}")

    for name, model in trained_models.items():
        # CHANGE: Use .joblib extension
        file_path = os.path.join(train_utils.MODEL_DIR, f"model_{name}.joblib")
        
        joblib.dump(model, file_path)
        print(f"   -> Saved to {file_path}")

    print("\nAll models saved successfully!")



if __name__ == "__main__":
    # 1. Get Data
    df = load_and_clean_data()

    # 2. Train (Returns the objects in memory)
    models = train_models(df)
    
    # 3. Save (Writes them to disk)
    save_models(models)