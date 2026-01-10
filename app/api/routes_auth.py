from fastapi import APIRouter
from pydantic import BaseModel
from app.core.security import create_token

router = APIRouter()

class AuthInput(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(auth_input: AuthInput):
    if auth_input.username == "admin" and auth_input.password == "password":
        token = create_token({"sub": auth_input.username})
        return {"access_token": token}
    return {"error": "Invalid credentials"}