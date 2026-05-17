
import joblib
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "app/ml/models/hybrid_trend_model.pkl"
OUTPUT_PATH = BASE_DIR / "app/ml/models/hybrid_model_params.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def export_model():
    print(f"Loading {MODEL_PATH}...")
    try:
        payload = joblib.load(MODEL_PATH)
        vec = payload['vectorizer']
        kmeans = payload['kmeans_model']
        
        # 1. Extract TF-IDF Stats
        # voc: word -> index
        vocab = vec.vocabulary_ 
        # idf: index -> weight
        idf = vec.idf_ 
        
        # 2. Extract Centroids
        # shape: (n_clusters, n_features)
        centroids = kmeans.cluster_centers_ 
        
        export_data = {
            "vocabulary": vocab,
            "idf": idf,
            "centroids": centroids,
            "n_clusters": kmeans.n_clusters
        }
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, cls=NumpyEncoder)
            
        print(f"✅ Successfully exported model params to {OUTPUT_PATH}")
        print(f"Vocab size: {len(vocab)}")
        print(f"Clusters: {len(centroids)}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export_model()
