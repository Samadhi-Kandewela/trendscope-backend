
print("--- Script Starting ---", flush=True)

import os
import joblib
import sys
from collections import Counter
from pathlib import Path

# Direct DB imports to bypass Flask App Context Overhead
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("--- Imports Done ---", flush=True)

# Configuration
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:samadhi@localhost:5432/trendscope_correct")
MODEL_DIR = Path("app/ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "hybrid_trend_model.pkl"
N_CLUSTERS = 200

def _build_text_simple(row):
    # Row is a tuple/dict from raw sql
    title = row['title_clean'] or row['title'] or ""
    tags = row['tags_clean'] or row['tags_text'] or ""
    desc = row['description_clean'] or ""
    if len(desc) > 500: desc = desc[:500]
    return f"{title} {tags} {desc}".strip()

def train_direct():
    print(f"--- Connecting to DB: {DB_URL} ---", flush=True)
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        print("--- Fetching Data ---", flush=True)
        # Fetch clean videos
        query = text("""
            SELECT title, title_clean, tags_text, tags_clean, description_clean, category_id, 'old' as source
            FROM videos_clean 
            LIMIT 5000
        """)
        result = conn.execute(query)
        rows = [dict(row._mapping) for row in result]
        print(f"Fetched {len(rows)} historical videos.", flush=True)

        # Fetch live videos
        query_live = text("""
            SELECT title, description, tags, category_id, 'live_api' as source, view_count
            FROM videos 
            WHERE source_dataset = 'live_api'
        """)
        result_live = conn.execute(query_live)
        
        for row in result_live:
            r = dict(row._mapping)
            # Map fields to match clean structure for text building
            r['title_clean'] = r['title']
            r['tags_text'] = " ".join(r['tags']) if r['tags'] else ""
            r['tags_clean'] = r['tags_text']
            r['description_clean'] = r['description']
            rows.append(r)
            
        print(f"Total rows: {len(rows)}", flush=True)
        
        if not rows:
            print("No data. Exiting.")
            return

        print("--- Generating TF-IDF Features ---", flush=True)
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 3)
        )
        texts = [_build_text_simple(r) for r in rows]
        embeddings = vectorizer.fit_transform(texts)
        
        print("--- Clustering ---", flush=True)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        print("--- Metadata ---", flush=True)
        cluster_meta = {}
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices: cluster_indices[label] = []
            cluster_indices[label].append(idx)
            
        for label, indices in cluster_indices.items():
            params = []
            for idx in indices:
                cat = rows[idx].get('category_id')
                if cat: params.append(str(cat)) # Simple category ID as proxy for genre for now
            
            dom = Counter(params).most_common(1)[0][0] if params else "General"
            # Map ID to genre string if possible, or just use ID for now
            # In real app we map ID -> Genre string. Here we just use "Category X"
            if dom.isdigit(): dom = f"Category {dom}"
            
            cluster_meta[label] = {"dominant_genre": dom, "size": len(indices)}

        print(f"--- Saving to {MODEL_PATH} ---", flush=True)
        payload = {
            "vectorizer": vectorizer,
            "kmeans_model": kmeans,
            "cluster_meta": cluster_meta
        }
        joblib.dump(payload, MODEL_PATH)
        print("Done.", flush=True)

if __name__ == "__main__":
    train_direct()
