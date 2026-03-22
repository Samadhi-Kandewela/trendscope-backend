
print("--- Script Starting ---", flush=True)

import os
import joblib
import numpy as np
from pathlib import Path
from collections import Counter
import sys

print("--- Imports Done ---", flush=True)

# from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("--- ML Imports Done ---", flush=True)

from app import create_app, db
from app.models.clean_video import CleanVideo
from app.models.video import Video # For live data
from app.services.trend_service import _build_text_for_clean_video

# Configuration
MODEL_DIR = Path("app/ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "hybrid_trend_model.pkl"

# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 200 # Back to 200 for good granularity with TF-IDF

def train_hybrid_model():
    print("--- Creating App ---", flush=True)
    app = create_app()
    with app.app_context():
        print("--- 1. Fetching Data ---", flush=True)
        # Fetch Clean Videos (Historical)
        # Increase limit back since TF-IDF is fast
        videos = db.session.query(CleanVideo).limit(5000).all() 
        print(f"Loaded {len(videos)} historical videos.", flush=True)
        
        # Fetch Live Videos (Recent)
        live_videos_raw = db.session.query(Video).filter(Video.source_dataset == 'live_api').all()
        print(f"Loaded {len(live_videos_raw)} live videos.", flush=True)

        # Convert live to CleanVideo format for consistency
        all_videos = list(videos)
        for lv in live_videos_raw:
            cv = CleanVideo()
            cv.id = 999999999 + int(lv.comment_count or 0) # Fake ID
            cv.video_id = lv.id
            cv.title = lv.title
            cv.title_clean = lv.title 
            cv.description_clean = lv.description
            cv.tags_clean = " ".join(lv.tags) if lv.tags else ""
            cv.view_count = lv.view_count
            cv.source_dataset = "live_api"
            cv.category_id = lv.category_id 
            # genre property maps category_id to genre name string
            cv.genre = lv.genre 
            all_videos.append(cv)

        print(f"Total training data: {len(all_videos)} videos.", flush=True)

        if not all_videos:
            print("No data found. Aborting.", flush=True)
            return

        print("--- 2. Generating Features (TF-IDF 1-3 N-Grams) ---", flush=True)
        # embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 3) # Capture phrases like "make money online"
        )
        texts = [_build_text_for_clean_video(v) for v in all_videos]
        embeddings = vectorizer.fit_transform(texts)
        
        print("--- 3. Clustering (K-Means) ---", flush=True)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        print("--- 4. Generating Metadata ---")
        # Calculate dominant genre per cluster
        cluster_meta = {}
        
        # Map labels to video indices
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)
            
        for label, indices in cluster_indices.items():
            # Get genres of videos in this cluster
            genres = []
            for idx in indices:
                v = all_videos[idx]
                # Try to get genre from explicit field or property
                g = getattr(v, 'genre', None)
                if not g and v.category_id:
                     # Fallback mapping if property missing on CleanVideo (it might be only on Video)
                     # But CleanVideo usually has genre string.
                     pass 
                if g:
                    genres.append(g)
            
            if not genres:
                dominant_genre = "General"
            else:
                dominant_genre = Counter(genres).most_common(1)[0][0]
                
            cluster_meta[label] = {
                "dominant_genre": dominant_genre,
                "size": len(indices)
            }
            
        print(f"Generated metadata for {len(cluster_meta)} clusters.", flush=True)

        print("--- 5. Saving Model ---", flush=True)
        payload = {
            "vectorizer": vectorizer, # Save entire TF-IDF object
            "kmeans_model": kmeans,
            "cluster_meta": cluster_meta
        }
        
        joblib.dump(payload, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}", flush=True)

if __name__ == "__main__":
    train_hybrid_model()
