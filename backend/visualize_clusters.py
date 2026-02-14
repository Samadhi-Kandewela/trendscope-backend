import sys
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Add backend to path to import app
sys.path.append(os.getcwd())

from app import create_app, db
from app.ml.inference import load_trend_topic_model
from app.models.clean_video import CleanVideo
from app.services.trend_service import _build_text_for_clean_video

def generate_cluster_visualization():
    app = create_app()
    with app.app_context():
        print("1. Loading Model...")
        model_payload = load_trend_topic_model()
        if not model_payload:
            print("Error: Model not found.")
            return

        vectorizer = model_payload["vectorizer"]
        kmeans = model_payload["kmeans"]

        print("2. Fetching Data...")
        # Get a sample of videos (limit 500 for cleaner plot)
        videos = db.session.query(CleanVideo).limit(500).all()
        texts = [_build_text_for_clean_video(v) for v in videos]
        
        print(f"3. Vectorizing {len(texts)} videos...")
        X = vectorizer.transform(texts)
        # Convert sparse matrix to dense for PCA
        X_dense = X.toarray()
        
        # Get Cluster Labels
        labels = kmeans.predict(X)

        print("4. Reducing Dimensions (PCA)...")
        # We use PCA for speed and "global structure"
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_dense)
        
        print("5. Plotting...")
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with colormap
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('Trend Topic Clusters (2D Projection)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path = os.path.join("app", "static", "cluster_structure.png")
        plt.savefig(output_path)
        print(f"Success! Image saved to: {output_path}")

if __name__ == "__main__":
    generate_cluster_visualization()
