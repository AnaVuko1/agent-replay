from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./agent_replay.db"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Security
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Features
    enable_websockets: bool = True
    max_context_snapshot_size: int = 10000  # characters
    enable_seed_data: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()