"""
Authentication module for JWT-based access control
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Security settings
SECRET_KEY = "your-secret-key-here-change-in-production"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# User storage file
USERS_FILE = Path("users.json")

class User(BaseModel):
    username: str
    hashed_password: str
    disabled: bool = False
    role: str = "user"  # ВАЖНО: добавлено поле role с дефолтным значением

class UserInDB(User):
    pass

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (temporary plain text comparison)"""
    # Temporary: plain text comparison for testing
    return plain_password == hashed_password

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def load_users() -> dict:
    """Load users - fixed for Render deployment"""
    # ВСЕГДА возвращаем дефолтных пользователей
    return {
        "admin": {
            "username": "admin",
            "hashed_password": "admin123",
            "disabled": False,
            "role": "admin"  # Явно указана роль admin
        },
        "analyst": {
            "username": "analyst",
            "hashed_password": "analyst123",
            "disabled": False,
            "role": "analyst"  # Явно указана роль analyst
        }
    }

def save_users(users: dict):
    """Save users to JSON file"""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user"""
    users = load_users()
    user_data = users.get(username)

    if not user_data:
        return None

    # Убедимся, что role есть в данных пользователя
    if "role" not in user_data:
        user_data["role"] = "user"  # Дефолтное значение

    user = UserInDB(**user_data)
    if not verify_password(password, user.hashed_password):
        return None

    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInDB:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    users = load_users()
    user_data = users.get(token_data.username)

    if user_data is None:
        raise credentials_exception

    # Убедимся, что role есть в данных пользователя
    if "role" not in user_data:
        user_data["role"] = "user"

    user = UserInDB(**user_data)

    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user