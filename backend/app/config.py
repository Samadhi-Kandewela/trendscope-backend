import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

def _normalize_db_url(url: str | None) -> str | None:
    """Ensure DB URL is SQLAlchemy-compatible."""
    if not url:
        return None
    # SQLAlchemy works fine with postgresql:// or postgresql+psycopg2://
    return url

# Load the RAPIDAPI_KEY
RAPIDAPI_KEY_ENV = os.getenv("RAPIDAPI_KEY", "")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")
    SQLALCHEMY_DATABASE_URI = _normalize_db_url(os.getenv("DB_URL"))
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # YouTube
    YOUTUBE_API_KEY = os.getenv("YT_API_KEY", "")

    # RAPIDAPI
    RAPIDAPI_KEY = RAPIDAPI_KEY_ENV

    # Simple knobs for trend logic (can tune later)
    GENRE_FORECAST_LOOKBACK_DAYS = int(os.getenv("GENRE_FORECAST_LOOKBACK_DAYS", 60))
    TREND_STRATEGY_LOOKBACK_DAYS = int(os.getenv("TREND_STRATEGY_LOOKBACK_DAYS", 90))

class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False
