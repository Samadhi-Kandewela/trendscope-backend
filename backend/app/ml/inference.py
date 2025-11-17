import os
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
from flask import current_app

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


_genre_forecast_model_cache: Optional[Dict[str, Any]] = None
_trend_topic_model_cache: Optional[Dict[str, Any]] = None


def load_genre_forecast_model() -> Optional[Dict[str, Any]]:
    """
    Loads genre_forecast_model.pkl once into memory.
    Returns a dict with keys: 'model', 'regions', 'genres'.
    """
    global _genre_forecast_model_cache
    if _genre_forecast_model_cache is not None:
        return _genre_forecast_model_cache

    model_path = MODELS_DIR / "genre_forecast_model.pkl"
    if not model_path.exists():
        current_app.logger.warning("Genre forecast model file not found at %s", model_path)
        return None

    _genre_forecast_model_cache = joblib.load(model_path)
    current_app.logger.info("Loaded genre forecast model from %s", model_path)
    return _genre_forecast_model_cache


def load_trend_topic_model() -> Optional[Dict[str, Any]]:
    """
    Loads trend_topic_model.pkl once into memory.
    Returns a dict with keys: 'vectorizer', 'kmeans', 'cluster_meta'.
    """
    global _trend_topic_model_cache
    if _trend_topic_model_cache is not None:
        return _trend_topic_model_cache

    model_path = MODELS_DIR / "trend_topic_model.pkl"
    if not model_path.exists():
        current_app.logger.warning("Trend topic model file not found at %s", model_path)
        return None

    _trend_topic_model_cache = joblib.load(model_path)
    current_app.logger.info("Loaded trend topic model from %s", model_path)
    return _trend_topic_model_cache
