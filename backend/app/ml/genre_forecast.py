from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from sqlalchemy import func

from .inference import load_genre_forecast_model
from ..extensions import db
from ..models.video import Video
from ..utils.genre_mapping import CATEGORY_TO_UI_GENRE


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


def _get_live_genre_distribution(region: str) -> Dict[str, float]:
    """
    Calculates the percentage of each genre in the Live API data (last 30 days).
    Returns {genre_name: share_float} (e.g. {"Gaming": 0.4, "Tech": 0.2})
    """
    # 1. Query Live Data
    query = db.session.query(Video.category_id).filter(
        Video.source_dataset == 'live_api'
    )
    
    if region != "Global":
        query = query.filter(Video.trending_country == region)
        
    # Get all category IDs
    rows = query.all()
    if not rows:
        return {}
        
    total = len(rows)
    counts = {}
    
    # 2. Map to UI Genres and Count
    for r in rows:
        cat_id = r[0]
        genre = CATEGORY_TO_UI_GENRE.get(cat_id, "Other")
        counts[genre] = counts.get(genre, 0) + 1
        
    # 3. Normalize to 0.0-1.0
    return {g: c / total for g, c in counts.items()}


def get_genre_forecast_distribution(
    region: str, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> List[Dict]:
    """
    Hybrid Forecast:
    1. Historical Model (Multi-Month Average)
    2. Live Data Adjustment (Real-time Correction)
    
    Formula: Final = (0.7 * Historical) + (0.3 * Live)
    """
    payload = load_genre_forecast_model()
    if payload is None:
        raise RuntimeError("Genre forecast model not available")

    regions = payload["regions"]
    genres = payload["genres"]
    
    # DEFAULT: Next 30 days if no dates provided
    if not start_date:
        start_date = datetime.utcnow()
    if not end_date:
        end_date = start_date + timedelta(days=30)

    # --- 1. HISTORICAL COMPONENT (Average over the months in range) ---
    # Identify unique months involved (e.g., Feb and March)
    months = set()
    curr = start_date
    while curr <= end_date:
        months.add(curr.month)
        # Advance by 15 days to efficiently cover months
        curr += timedelta(days=15)
        
    historical_accum = {g: 0.0 for g in genres}
    
    for m in months:
        if region == "Global":
            # Global = Average of all support regions
            month_dist = {g: 0.0 for g in genres}
            for r in regions:
                r_dist = _predict_for_single_region(r, m)
                for g, v in r_dist.items():
                    month_dist[g] += v
            # Normalize global for this month
            n_regions = len(regions) or 1
            for g in month_dist:
                month_dist[g] /= n_regions
        else:
            # Specific Region
            month_dist = _predict_for_single_region(region, m)
            
        # Add to total accumulator
        for g, v in month_dist.items():
            historical_accum[g] += v
            
    # Average over number of months
    n_months = len(months) or 1
    historical_avg = {g: v / n_months for g, v in historical_accum.items()}
    
    # --- 2. LIVE COMPONENT (Real-time Reality Check) ---
    live_dist = _get_live_genre_distribution(region)
    
    # --- 3. HYBRID FUSION ---
    # Weight: 70% History (Stable), 30% Live (Reactive)
    # If no live data, use 100% History
    
    final_dist = {}
    has_live = len(live_dist) > 0
    alpha = 0.3 if has_live else 0.0
    
    for g in genres:
        hist_score = historical_avg.get(g, 0.0)
        live_score = live_dist.get(g, 0.0) # Default 0 if genre not in live data
        
        final_score = ((1 - alpha) * hist_score) + (alpha * live_score)
        final_dist[g] = final_score

    # Convert to sorted list
    results = [
        {"name": g, "percentage": round(final_dist[g] * 100.0, 1)}
        for g in genres
    ]

    results.sort(key=lambda x: x["percentage"], reverse=True)
    return results
