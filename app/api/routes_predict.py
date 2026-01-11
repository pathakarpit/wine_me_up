from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.dependencies import get_api_key, get_current_user
from app.services.model_services import predict_wine_quality # Ensure import matches filename

router = APIRouter()

# 1. Define the Input Schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# 2. Update the Route
# We add {model_name} to the path so it matches "/predict/xgboost"
@router.post('/predict/{model_name}')
def predict_quality(
    model_name: str,              # <--- Capture model name from URL
    wine: WineFeatures,           # <--- Capture features from JSON Body
    user=Depends(get_current_user),
    _=Depends(get_api_key)
):
    try:
        # 3. Pass both arguments to the service
        prediction = predict_wine_quality(wine.model_dump(), model_name)
        
        return {
            "model_used": model_name,
            "predicted_quality": prediction
        }
    except ValueError as e:
        # Handle case where model_name isn't found
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))