
import joblib
from pathlib import Path
import sys

MODEL_PATH = Path("app/ml/models/hybrid_trend_model.pkl")

print(f"--- Attempting to load {MODEL_PATH} ---")

try:
    if not MODEL_PATH.exists():
        print("❌ File does not exist")
        sys.exit(1)

    print(f"File size: {MODEL_PATH.stat().st_size} bytes")
    
    payload = joblib.load(MODEL_PATH)
    print("✅ Load Successful")
    print(f"Keys: {list(payload.keys())}")
    
    vec = payload.get('vectorizer')
    kmeans = payload.get('kmeans_model')
    
    print(f"Vectorizer Type: {type(vec)}")
    print(f"KMeans Type: {type(kmeans)}")
    
    if hasattr(kmeans, 'n_clusters'):
        print(f"Clusters: {kmeans.n_clusters}")
        
except Exception as e:
    print(f"❌ CRASH/ERROR: {e}")
    import traceback
    traceback.print_exc()
