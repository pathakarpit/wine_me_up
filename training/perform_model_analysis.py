import pandas as pd
import numpy as np
import joblib
import os
import sys
from training.train_utils import DATA_FILE_PATH, MODEL_DIR, EDA_DIR, EMA_PATH
import warnings
from typing import Dict, Any

# Metric & Selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Optimization
import optuna

# Mute warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        if 'Id' in df.columns: df = df.drop('Id', axis=1)
        df['quality'] = df['quality'].astype(int)
        return df
    except FileNotFoundError:
        sys.exit(f"❌ Error: Data file not found at {DATA_FILE_PATH}")

# ==========================================
# STEP 1: VERIFY STRATIFICATION
# ==========================================
def verify_stratification_step(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- Step 1: Verifying Data Split Stratification ---")
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(X, y))
    
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    dist_df = pd.DataFrame({
        'Train Set %': y_train.value_counts(normalize=True).sort_index(),
        'Test Set %': y_test.value_counts(normalize=True).sort_index()
    })
    
    return {
        'title': "1. Stratification Verification",
        'type': 'stratification_plot',
        'data': {'distribution': dist_df},
        'comment': "Proving that Stratified K-Fold preserved class ratios.",
        'code': "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
    }

# ==========================================
# STEP 2: BASELINE TOURNAMENT (Pre-Tuned)
# ==========================================
def run_baseline_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- Step 2: Baseline Tournament (Default Params) ---")
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "SVC": SVC(random_state=42)
    }

    results = []

    # A. Classifiers
    for name, model in classifiers.items():
        fold_kappas = []
        y_target = (y - 3) if name == "XGBoost" else y
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
            
            model.fit(X_train, y_train)
            if name == "XGBoost":
                preds = model.predict(X_test)
                fold_kappas.append(cohen_kappa_score(y_test, preds, weights='quadratic'))
            else:
                preds = model.predict(X_test)
                fold_kappas.append(cohen_kappa_score(y_test, preds, weights='quadratic'))
        
        results.append({"Model": name, "Kappa Score": np.mean(fold_kappas)})
        print(f"   -> {name}: {np.mean(fold_kappas):.4f}")

    # B. Regressor Strategy
    print("   -> Running LGBM Regressor Strategy...")
    reg_kappas = []
    reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        reg.fit(X_train, y_train)
        preds = np.round(reg.predict(X_test)).astype(int)
        reg_kappas.append(cohen_kappa_score(y_test, preds, weights='quadratic'))
        
    results.append({"Model": "LGBM Regressor (Rounded)", "Kappa Score": np.mean(reg_kappas)})

    return {
        'title': "2. Baseline Model Leaderboard (Pre-Tuned)",
        'type': 'model_leaderboard',
        'data': {'results': pd.DataFrame(results).sort_values(by="Kappa Score", ascending=False)},
        'comment': "A fair comparison of models using their default parameters.",
        'code': "# ... Baseline Loop Code ..."
    }

# ==========================================
# STEP 3: TUNED TOURNAMENT (Optuna)
# ==========================================
def objective_factory(trial, model_name, X, y, skf):
    params = {}
    model = None
    
    # --- RANDOM FOREST ---
    # CHANGED: Increased max_depth (20 -> 50) and n_estimators (300 -> 500)
    if model_name == "Random Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500), 
            'max_depth': trial.suggest_int('max_depth', 10, 50),         
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'random_state': 42
        }
        model = RandomForestClassifier(**params)

    # --- XGBOOST ---
    # CHANGED: Increased depth and subsample ranges
    elif model_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 15), 
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        
    # --- LIGHTGBM ---
    elif model_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150), 
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        
    # --- CATBOOST ---
    elif model_name == "CatBoost":
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': 42,
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
        
    # --- SVC ---
    # CHANGED: Widened C range significantly
    elif model_name == "SVC":
        params = {
            'C': trial.suggest_float('C', 0.5, 50.0, log=True), 
            'kernel': 'rbf',
            'random_state': 42
        }
        model = SVC(**params)

    # --- REGRESSOR STRATEGY ---
    elif model_name == "LGBM Regressor (Rounded)":
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)

    # --- EVALUATION ---
    kappas = []
    y_target = (y - 3) if model_name == "XGBoost" else y

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        if "Regressor" in model_name:
            final_preds = np.round(model.predict(X_test)).astype(int)
            y_eval = y.iloc[test_idx]
        elif model_name == "XGBoost":
            final_preds = model.predict(X_test) + 3
            y_eval = y.iloc[test_idx]
        else:
            final_preds = model.predict(X_test)
            y_eval = y_test

        kappas.append(cohen_kappa_score(y_eval, final_preds, weights='quadratic'))
        
    return np.mean(kappas)

def run_tuned_tournament(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- Step 3: Tuned Tournament (Optuna - 20 Trials) ---")
    X, y = df.drop('quality', axis=1), df['quality']
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    results = []
    model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVC", "LGBM Regressor (Rounded)"]
    
    for name in model_names:
        print(f"   -> Tuning {name}...")
        
        # CHANGED: Increased n_trials from 5 to 20
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective_factory(t, name, X, y, skf), n_trials=20) 
        
        print(f"      Best Kappa: {study.best_value:.4f}")
        
        results.append({
            "Model": name,
            "Best Kappa Score": study.best_value,
            "Best Params": study.best_params
        })

    return {
        'title': "3. Final Leaderboard (Hyperparameter Tuned)",
        'type': 'tuned_leaderboard',
        'data': {'results': pd.DataFrame(results).sort_values(by="Best Kappa Score", ascending=False)},
        'comment': "Performance after giving every model a fair chance via Bayesian Optimization (20 Trials).",
        'code': "# Optuna Loop..."
    }

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
def generate_model_artifacts():
    print("--- Starting Advanced Model Analysis ---")
    df = load_data()
    artifacts: Dict[str, Any] = {'model_sections': []}

    artifacts['model_sections'].append(verify_stratification_step(df))
    artifacts['model_sections'].append(run_baseline_comparison(df)) # Step 2
    artifacts['model_sections'].append(run_tuned_tournament(df))    # Step 3

    if not os.path.exists(EDA_DIR): os.makedirs(EDA_DIR)
    joblib.dump(artifacts, EMA_PATH)
    print(f"\n✅ Model Artifacts saved to: {EMA_PATH}")

if __name__ == "__main__":
    generate_model_artifacts()