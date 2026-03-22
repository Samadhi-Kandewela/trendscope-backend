
import os
import joblib
from pathlib import Path
from collections import Counter
from flask.cli import with_appcontext
import click

def register_commands(app):
    
    @app.cli.command("train-hybrid")
    def train_hybrid():
        """Train the Hybrid Trend Model (TF-IDF + K-Means)"""
        print("--- CLI: Starting Hybrid Training ---", flush=True)
        
        # Imports inside function to avoid top-level hang
        try:
            print("Importing ML libraries...", flush=True)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from .models.clean_video import CleanVideo
            from .models.video import Video
            from .extensions import db
            from .services.trend_service import _build_text_for_clean_video
            print("Imports successful.", flush=True)
        except Exception as e:
            print(f"Import Error: {e}", flush=True)
            return

        # Configuration
        MODEL_DIR = Path("app/ml/models")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH = MODEL_DIR / "hybrid_trend_model.pkl"
        N_CLUSTERS = 200

        print("--- 1. Fetching Data ---", flush=True)
        videos = db.session.query(CleanVideo).limit(5000).all()
        print(f"Loaded {len(videos)} historical videos.", flush=True)
        
        live_videos_raw = db.session.query(Video).filter(Video.source_dataset == 'live_api').all()
        print(f"Loaded {len(live_videos_raw)} live videos.", flush=True)

        all_videos = list(videos)
        for lv in live_videos_raw:
            cv = CleanVideo()
            cv.id = 999999999 + int(lv.comment_count or 0)
            cv.video_id = lv.id
            cv.title = lv.title
            cv.title_clean = lv.title 
            cv.description_clean = lv.description
            cv.tags_clean = " ".join(lv.tags) if lv.tags else ""
            cv.view_count = lv.view_count
            cv.source_dataset = "live_api"
            cv.category_id = lv.category_id 
            cv.genre = lv.genre 
            all_videos.append(cv)

        print(f"Total training data: {len(all_videos)} videos.", flush=True)

        if not all_videos:
            print("No data found. Aborting.", flush=True)
            return

        print("--- 2. Generating Features (TF-IDF 1-3 N-Grams) ---", flush=True)
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 3) 
        )
        texts = [_build_text_for_clean_video(v) for v in all_videos]
        embeddings = vectorizer.fit_transform(texts)
        
        print("--- 3. Clustering (K-Means) ---", flush=True)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        print("--- 4. Generating Metadata ---", flush=True)
        cluster_meta = {}
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices: cluster_indices[label] = []
            cluster_indices[label].append(idx)
            
        for label, indices in cluster_indices.items():
            genres = []
            for idx in indices:
                v = all_videos[idx]
                g = getattr(v, 'genre', None)
                if g: genres.append(g)
            
            if not genres:
                dom = "General"
            else:
                dom = Counter(genres).most_common(1)[0][0]
                
            cluster_meta[label] = {
                "dominant_genre": dom,
                "size": len(indices)
            }
            
        print(f"Generated metadata for {len(cluster_meta)} clusters.", flush=True)

        print("--- 5. Saving Model ---", flush=True)
        payload = {
            "vectorizer": vectorizer,
            "kmeans_model": kmeans,
            "cluster_meta": cluster_meta
        }
        
        joblib.dump(payload, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}", flush=True)
