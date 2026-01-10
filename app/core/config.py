import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "wine_me_up"
    API_KEY: str = os.getenv("API_KEY", "safe-default-key")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256" 
    REDIS_URL: str = os.getenv("REDIS_URL")
    MODEL_DIR: str = "app/models"
    MODEL_NAME: str = "default_model.joblib"
    MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_NAME)

    @classmethod
    def set_model_name(cls, name: str) -> None:
        """Set model name at runtime (call from your app)."""
        cls.MODEL_NAME = name
        cls.MODEL_PATH = os.path.join(cls.MODEL_DIR, cls.MODEL_NAME)

settings = Settings()


"""
from app.core.config import settings

settings.set_model_name("dummy_model.joblib")
"""