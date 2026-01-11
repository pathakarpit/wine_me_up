import os

# Root Data Directory (Raw CSVs)
DATA_DIR = 'data'
DATA_FILE_NAME = 'WineQT.csv'
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# App Directory Structure
APP_DIR = 'app'

# Models
MODEL_DIR_NAME = 'models'
MODEL_DIR = os.path.join(APP_DIR, MODEL_DIR_NAME)
MODEL_NAME = 'default_model.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Artifacts (EDA & Model Analysis)
EDA_DIR = os.path.join(APP_DIR, 'data') # app/data
EDA_PATH = os.path.join(EDA_DIR, 'eda_artifacts.joblib')
EMA_PATH = os.path.join(EDA_DIR, 'ema_artifacts.joblib') # Experimental Model Analysis