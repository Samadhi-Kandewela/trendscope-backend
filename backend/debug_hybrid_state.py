
import logging
import joblib
from pathlib import Path
from flask import Flask
from app import create_app, db
from app.models.clean_video import CleanVideo
from app.services.trend_service import _build_text_for_clean_video
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("app/ml/models")
MODEL_PATH = MODELS_DIR / "hybrid_trend_model.pkl"

def check_hybrid_state():
    print("--- CHECKING HYBRID MODEL STATE ---")
    
    # 1. Check if model file exists
    if MODEL_PATH.exists():
        print(f"✅ Model file exists at: {MODEL_PATH}")
        try:
            payload = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully.")
            print(f"Payload Keys: {list(payload.keys())}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    else:
        print(f"❌ Model file NOT found at: {MODEL_PATH}")
        
    # 2. Try Manual Training (Simulate On-the-Fly)
    print("\n--- SIMULATING TRAINING ---")
    app = create_app()
    with app.app_context():
        try:
            print("Fetching sample data...", flush=True)
            data = db.session.query(CleanVideo).limit(100).all()
            if not data:
                print("❌ No data found in database.")
                return
            
            print(f"Found {len(data)} videos.", flush=True)
            
            print("Vectorizing...", flush=True)
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            texts = [_build_text_for_clean_video(v) for v in data]
            embeddings = vectorizer.fit_transform(texts)
            
            print("Clustering...", flush=True)
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=5)
            labels = kmeans.fit_predict(embeddings)
            
            print("✅ SUCCEEDED: Training logic works.")
            
            # Try saving just to test permissions
            payload = {
                "vectorizer": vectorizer,
                "kmeans_model": kmeans,
                "cluster_meta": {}
            }
            joblib.dump(payload, MODEL_PATH)
            print(f"✅ SUCCEEDED: Model saved to {MODEL_PATH}")
            
        except Exception as e:
            print(f"❌ FAILED: Training simulation error: {e}")

if __name__ == "__main__":
    check_hybrid_state()
