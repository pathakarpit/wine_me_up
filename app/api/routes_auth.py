from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.core.security import create_token

router = APIRouter()

class AuthInput(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(auth_input: AuthInput):
    """
    Authenticates a user and returns a JWT token.
    """
    # 1. Verify Credentials (dummy check)
    if auth_input.username == "admin" and auth_input.password == "password":
        
        # 2. Create Token
        # We explicitly name the argument 'data' for clarity
        token = create_token(data={"sub": auth_input.username})
        
        # 3. Return Standard OAuth2 Response
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    
    # 4. Handle Failure properly with 401 Status Code
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )