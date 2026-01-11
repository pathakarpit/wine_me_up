import joblib
import pandas as pd
import os
import json
import hashlib
from app.core.config import settings
from app.cache.redis_cache import get_cached_prediction, set_cached_prediction

# 1. Global In-Memory Cache for Loaded Models
# This prevents reading from disk on every single request.
loaded_models = {}

def get_model_path(model_name: str) -> str:
    """
    Safely resolves the file path for a requested model.
    """
    # Map API names to actual filenames on disk
    filename_map = {
        "default": "default_model.joblib", # Usually maps to best performer
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost.joblib",
        "lightgbm_classifier": "lightgbm_classifier.joblib",
        "catboost": "catboost.joblib",
        "svc": "svc.joblib",
        "lightgbm_regressor": "lightgbm_regressor.joblib" # The Secret Weapon
    }
    
    filename = filename_map.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return os.path.join(settings.MODEL_DIR, filename)

def generate_cache_key(data: dict, model_name: str) -> str:
    """
    Creates a unique, deterministic fingerprint for the request.
    Key = Model Name + Hash of Input Data
    """
    # Sort keys ensures {"alcohol":10, "pH":3} == {"pH":3, "alcohol":10}
    data_str = json.dumps(data, sort_keys=True)
    
    # Create MD5 hash of inputs
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    
    return f"predict:{model_name}:{data_hash}"

def predict_wine_quality(data: dict, model_name: str):
    """
    Main service function: Check Cache -> Load Model -> Predict -> Save Cache
    """
    
    # --- STEP 1: CHECK REDIS CACHE ---
    try:
        cache_key = generate_cache_key(data, model_name)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            print(f"⚡ Cache Hit: {model_name}")
            return cached_result['predicted_quality']
    except Exception as e:
        print(f"⚠️ Redis Error (Proceeding without cache): {e}")

    # --- STEP 2: LOAD MODEL (LAZY LOADING) ---
    # We only load the model if it's not already in RAM.
    if model_name not in loaded_models:
        try:
            model_path = get_model_path(model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file missing: {model_path}")
            
            print(f"🔄 Loading Model into Memory: {model_name}...")
            loaded_models[model_name] = joblib.load(model_path)
            
        except Exception as e:
            # If specific model fails, fallback to default? Or raise error?
            # Here we raise error to let the user know.
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
    
    model = loaded_models[model_name]

    # --- STEP 3: PREPARE DATA ---
    # Convert input dict to DataFrame (ensures column names match training)
    input_df = pd.DataFrame([data])
    
    column_mapping = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides", # No change needed, but kept for clarity
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol"
    }
    input_df = input_df.rename(columns=column_mapping)
    # --- STEP 4: PREDICT ---
    try:
        prediction_output = model.predict(input_df)[0]
        raw_value = float(prediction_output)
        # Logic: If it's a Regressor (Secret Weapon), we round the float.
        # If it's a Classifier, it returns an int directly, but round() is safe for both.
        final_quality = int(round(raw_value))

    except Exception as e:
        raise RuntimeError(f"Prediction failed during inference: {e}")

    # --- STEP 5: SAVE TO REDIS ---
    try:
        cache_payload = {
            "model": model_name,
            "predicted_quality": final_quality,
            "raw_value": float(raw_prediction) # Optional: nice for debugging
        }
        set_cached_prediction(cache_key, cache_payload)
    except Exception as e:
        print(f"⚠️ Failed to write to Redis: {e}")

    return final_quality