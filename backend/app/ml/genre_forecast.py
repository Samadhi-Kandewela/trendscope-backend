# backend/app/ml/genre_forecast.py

from datetime import datetime
from typing import List, Dict

import numpy as np

from .inference import load_genre_forecast_model


def _predict_for_single_region(region_code: str, month: int) -> Dict[str, float]:
    """
    Predict raw genre shares for one region & month using the trained model.
    Returns {genre_name: share_float}.
    """
    payload = load_genre_forecast_model()
    if payload is None:
        raise RuntimeError("Genre forecast model not loaded. Did you train it?")

    model = payload["model"]
    regions = payload["regions"]
    genres = payload["genres"]

    # Build feature vector: [month, region_one_hot...]
    region_vec = [0.0] * len(regions)
    if region_code in regions:
        idx = regions.index(region_code)
        region_vec[idx] = 1.0

    X = np.array([[month] + region_vec], dtype=float)
    y = model.predict(X)[0]  # array of shares per genre

    # Clean & normalize
    y = np.maximum(y, 0.0)
    s = float(y.sum())
    if s > 0:
        y = y / s

    return {g: float(v) for g, v in zip(genres, y)}


def get_genre_forecast_distribution(region: str, days: int = 30) -> List[Dict]:
    """
    Public helper used by trend_service.

    Input:
      region: "US", "IN", ... or "Global"
      days: forecast horizon (not directly used yet, but kept for future)

    Output:
      [
        {"name": "Tech", "percentage": 34.2},
        {"name": "Gaming", "percentage": 28.7},
        ...
      ]
    """
    payload = load_genre_forecast_model()
    if payload is None:
        raise RuntimeError("Genre forecast model not available")

    regions = payload["regions"]
    genres = payload["genres"]

    # Use current month as proxy for "near term"
    month = datetime.utcnow().month

    if region == "Global":
        # Average prediction over all regions
        accum = {g: 0.0 for g in genres}
        for r in regions:
            dist_r = _predict_for_single_region(r, month)
            for g, v in dist_r.items():
                accum[g] += v

        # Average
        n = float(len(regions)) or 1.0
        for g in accum:
            accum[g] /= n
    else:
        accum = _predict_for_single_region(region, month)

    # Convert to sorted list
    results = [
        {"name": g, "percentage": round(accum[g] * 100.0, 1)}
        for g in genres
    ]

    results.sort(key=lambda x: x["percentage"], reverse=True)
    return results
