
import sys
import json
import joblib
import traceback
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Setup Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "app/ml/models/hybrid_trend_model.pkl"

def run_inference(video_data):
    """
    Input: List of dicts with 'title_clean', 'tags', 'description_clean'
    Output: List of cluster labels
    """
    try:
        # 1. Load Model
        if not MODEL_PATH.exists():
            return {"error": "Model file missing"}
            
        payload = joblib.load(MODEL_PATH)
        vectorizer = payload.get("vectorizer")
        kmeans = payload.get("kmeans_model")
        
        if not vectorizer or not kmeans:
            return {"error": "Corrupt model payload"}

        # 2. Prepare Text
        texts = []
        for v in video_data:
            t = (v.get('title_clean') or "").strip()
            TAG = (v.get('tags_clean') or "").strip()
            d = (v.get('description_clean') or "").strip()
            if len(d) > 500: d = d[:500]
            texts.append(f"{t} {TAG} {d}".strip())

        if not texts:
             return {"labels": []}

        # 3. Predict
        embeddings = vectorizer.transform(texts)
        labels = kmeans.predict(embeddings)
        
        # 4. Return
        return {"labels": labels.tolist()}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    try:
        if len(sys.argv) >= 3:
            # File Mode
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            result = run_inference(data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        else:
            # Stdin Mode (Fallback)
            input_str = sys.stdin.read()
            if not input_str:
                print(json.dumps({"error": "No input received"}))
                sys.exit(1)
                
            data = json.loads(input_str)
            result = run_inference(data)
            print(json.dumps(result))

    except Exception as e:
        err_msg = json.dumps({"error": f"Script failed: {e}"})
        if len(sys.argv) >= 3:
             with open(sys.argv[2], 'w') as f:
                 f.write(err_msg)
        else:
            print(err_msg)
        sys.exit(1)
