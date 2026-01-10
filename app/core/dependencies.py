from fastapi import Header, HTTPException
from app.core.config import settings
from app.core.security import verify_token

def get_api_key(api_key: str = Header(...)) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

def get_current_user(token: str = Header(...)) -> dict:
    # robustly handle "Bearer <token>" or just "<token>"
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return payload