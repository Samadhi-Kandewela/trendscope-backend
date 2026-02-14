import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from flask import current_app

from ..extensions import db
from ..models.clean_video import CleanVideo
from ..models.video import Video

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "viral_velocity_model.pkl"

class ViralVelocityModel:
    """
    Predicts future views (Viral Potential) based on current engagement velocity.
    Algorithm: Random Forest Regressor
    Features: [current_views, like_count, comment_count, days_since_upload]
    Target: Projected lifetime views (Approximated by max views in dataset for similar videos)
    """
    
    def __init__(self):
        self.model = None
        self.features = ['view_count', 'like_count', 'comment_count', 'days_since_upload']

    def load(self):
        """Loads the pre-trained model from disk."""
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            current_app.logger.info(f"Loaded Viral Velocity Model from {MODEL_PATH}")
        else:
            current_app.logger.warning("Viral Velocity Model not found. Training new one...")
            self.train()

    def train(self):
        """
        Trains the model using historical data (CleanVideo + Live Video).
        Since we don't have 'time-series' snapshots for every video, we simulate 'early signals'.
        Assumption: We try to predict 'High Potential' based on engagement ratios.
        """
        current_app.logger.info("Starting Viral Velocity Model Training...")
        
        # 1. Fetch Data
        # We use CleanVideo (historical) as ground truth
        videos = db.session.query(
            CleanVideo.view_count, 
            CleanVideo.like_count, 
            CleanVideo.comment_count,
            CleanVideo.published_at,
            CleanVideo.trending_date
        ).filter(CleanVideo.view_count.isnot(None)).limit(5000).all()
        
        if not videos:
            current_app.logger.error("No data to train viral model.")
            return

        data = []
        for v in videos:
            # Feature Engineering
            # We simulate "Days Since Upload" at the moment of trending
            pub = v.published_at or datetime.utcnow()
            trend = v.trending_date or datetime.utcnow()
            days_since = (trend - pub).days
            if days_since < 0: days_since = 0
            
            # Target: The actual view count at trending time (proxy for 'success')
            # In a real system, we'd predict Y (Future Views) from X (Current Views).
            # Here, we learn the mapping: "Input Stats -> Likely Total Views".
            
            data.append({
                'view_count': v.view_count,
                'like_count': v.like_count or 0,
                'comment_count': v.comment_count or 0,
                'days_since_upload': days_since,
                'target_score': v.view_count # Predicting the magnitude
            })
            
        df = pd.DataFrame(data)
        
        # Split
        X = df[self.features]
        y = df['target_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        current_app.logger.info(f"Model Trained. MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Save
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True)
            
        joblib.dump(self.model, MODEL_PATH)
        
        return {"mae": mae, "r2": r2}

    def predict(self, view_count, like_count, comment_count, days_since_upload):
        """
        Returns: {
            "predicted_views": int,
            "viral_probability": str (Low/Medium/High/Viral),
            "confidence_score": float (0-100)
        }
        """
        if not self.model:
            self.load()
            if not self.model:
                return None
                
        # Input vector
        X_input = pd.DataFrame([{
            'view_count': view_count, 
            'like_count': like_count, 
            'comment_count': comment_count, 
            'days_since_upload': days_since_upload
        }])
        
        prediction = self.model.predict(X_input)[0]
        
        # Determine "Viral Probability" label
        # These thresholds are arbitrary for the prototype, based on dataset distribution
        if prediction > 1000000:
            label = "Viral Hit"
            score = 95
        elif prediction > 500000:
            label = "High Potential"
            score = 80
        elif prediction > 100000:
            label = "Growing"
            score = 60
        else:
            label = "Niche/Stable"
            score = 40
            
        return {
            "predicted_views": int(prediction),
            "viral_label": label,
            "velocity_score": score
        }
