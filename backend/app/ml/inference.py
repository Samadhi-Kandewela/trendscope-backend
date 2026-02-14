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


def calculate_clustering_metrics(X, labels) -> float:
    """
    Calculates the Silhouette Score for the given data and labels.
    Range: -1 (Incorrect) to +1 (Highly Dense/Separated).
    0 means overlapping clusters.
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # Silhouette requires at least 2 labels and < N samples
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= X.shape[0]:
        return 0.0
        
    # Limit sample size for performance if needed (e.g. > 2000 samples)
    if X.shape[0] > 2000:
        from sklearn.utils import resample
        X_sample, labels_sample = resample(X, labels, n_samples=2000, random_state=42)
        score = silhouette_score(X_sample, labels_sample)
    else:
        score = silhouette_score(X, labels)
        
    return float(score)


# NEW (Phase 6): Advanced S-BERT + UMAP Model Loader
_advanced_model_cache: Optional[Dict[str, Any]] = None

def load_advanced_trend_model() -> Optional[Dict[str, Any]]:
    """
    Loads advanced_trend_model.pkl (S-BERT + UMAP + HDBSCAN)
    """
    global _advanced_model_cache
    if _advanced_model_cache is not None:
        return _advanced_model_cache

    model_path = MODELS_DIR / "advanced_trend_model.pkl"
    if not model_path.exists():
        current_app.logger.warning("Advanced model file not found at %s", model_path)
        return None

    try:
        _advanced_model_cache = joblib.load(model_path)
        current_app.logger.info("Loaded ADVANCED trend model from %s", model_path)
        return _advanced_model_cache
    except Exception as e:
        current_app.logger.error("Failed to load advanced model: %s", e)
        return None
