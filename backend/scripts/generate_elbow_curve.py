import os
import sys
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo
from scripts.train_trend_topic_model import CUSTOM_STOPWORDS_LIST

app = create_app()

def generate_elbow_curve():
    print("Fetching sample videos for Elbow Curve generation...")
    with app.app_context():
        # Fetch a representative sample to make generation fast but accurate
        videos = db.session.query(CleanVideo).filter(CleanVideo.title_clean.isnot(None)).order_by(CleanVideo.view_count.desc().nullslast()).limit(10000).all()
        
        docs = []
        for v in videos:
            title = (v.title_clean or v.title or "").strip()
            tags = (v.tags_clean or v.tags_text or "").strip()
            # Omit description to speed up TF-IDF for the graph
            docs.append(f"{title} {tags}".strip())
            
        print(f"Prepared {len(docs)} documents.")
        
        vectorizer = TfidfVectorizer(
            max_features=10000, 
            min_df=5,
            max_df=0.5,
            ngram_range=(1, 2),
            stop_words=CUSTOM_STOPWORDS_LIST
        )
        
        X = vectorizer.fit_transform(docs)
        print(f"TF-IDF matrix shape: {X.shape}")
        
        ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        inertias = []
        
        for k in ks:
            print(f"Training K={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
        print("Rendering graph...")
        plt.figure(figsize=(10, 6))
        plt.plot(ks, inertias, 'go-', linewidth=2, markersize=8)
        plt.title('Elbow Method for Optimal K (K-Means Topic Modeling)', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
        
        # Highlight K=30
        plt.axvline(x=30, color='r', linestyle='--', label='Optimal K (30)')
        plt.scatter(30, inertias[ks.index(30)], color='red', s=100, zorder=5)
        plt.annotate('Elbow Point\n(Diminishing Returns)',
                     xy=(30, inertias[ks.index(30)]),
                     xytext=(35, inertias[ks.index(30)] + min(inertias)*0.01),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10)
                     
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save straight to artifacts
        out_path = r"C:\Users\Dell\.gemini\antigravity\brain\df0d6884-9f31-4396-ab89-73ed204df10a\elbow_curve.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved elbow curve graph to {out_path}")

if __name__ == "__main__":
    generate_elbow_curve()
