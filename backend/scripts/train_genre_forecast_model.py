"""
Train a model to forecast genre distribution per region & time
using cleaned data from videos_clean (CleanVideo).

Usage:
    python -m scripts.train_genre_forecast_model
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo  # <- from videos_clean

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib


def build_monthly_region_genre_matrix():
    """
    Build training data from videos_clean:

    For each (region, month) bucket, we compute the share of each genre.

    Output:
      X: 2D numpy array of features per sample
         [month (1–12), region_one_hot...]
      Y: 2D numpy array of genre share per sample
      regions: list of region codes (['US', 'IN', 'GB', ...])
      genres: list of UI genres (['Tech', 'Gaming', 'Lifestyle', ...])
    """

    rows = db.session.query(
        CleanVideo.trending_country_code,
        CleanVideo.trending_date,
        CleanVideo.genre,
    ).filter(
        CleanVideo.trending_date.isnot(None),
        CleanVideo.trending_country_code.isnot(None),
        CleanVideo.genre.isnot(None),
    ).all()

    if not rows:
        raise RuntimeError("No data found in videos_clean for training genre forecast model.")

    # (region, month) -> {genre: count}
    buckets = defaultdict(lambda: defaultdict(int))

    for region_code, trending_date, genre in rows:
        if not region_code or not trending_date or not genre:
            continue

        month = trending_date.month  # 1–12
        key = (region_code, month)
        buckets[key][genre] += 1

    regions = sorted({region for (region, _month) in buckets.keys()})
    genres = sorted({g for counts in buckets.values() for g in counts.keys()})

    region_to_idx = {r: i for i, r in enumerate(regions)}

    X = []
    Y = []

    for (region, month), counts in buckets.items():
        # region one-hot
        region_vec = [0] * len(regions)
        region_vec[region_to_idx[region]] = 1

        features = [month] + region_vec

        total = sum(counts.values()) or 1
        shares = [counts.get(g, 0) / total for g in genres]

        X.append(features)
        Y.append(shares)

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    return X, Y, regions, genres


def train_and_save_model():
    """
    Train RandomForest-based MultiOutputRegressor and save payload:

      {
        "model": model,
        "regions": regions,
        "genres": genres,
      }

    This matches what app/ml/inference.py expects.
    """
    app = create_app()
    with app.app_context():
        X, Y, regions, genres = build_monthly_region_genre_matrix()

        app.logger.info("Training genre forecast model on %d samples", X.shape[0])
        app.logger.info("Regions: %s", regions)
        app.logger.info("Genres: %s", genres)

        base_reg = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base_reg)
        model.fit(X, Y)

        models_dir = Path("app/ml/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": model,
            "regions": regions,
            "genres": genres,
        }

        out_path = models_dir / "genre_forecast_model.pkl"
        joblib.dump(payload, out_path)
        app.logger.info("Saved genre forecast model to %s", out_path)


if __name__ == "__main__":
    train_and_save_model()
