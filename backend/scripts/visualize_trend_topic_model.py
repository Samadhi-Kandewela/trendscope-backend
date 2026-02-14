"""
Visualize the trained trend topic model (TF-IDF + KMeans).

This script:
  - Loads app/ml/models/trend_topic_model.pkl
  - Rebuilds documents from CleanVideo (same style as training)
  - Predicts cluster labels for a subset of videos
  - Plots:
      1) Top terms (by TF-IDF weight in cluster center)
      2) Genre distribution inside a chosen cluster

Usage (from backend folder):
    python -m scripts.visualize_trend_topic_model
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import joblib

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo


# --------------------------------------------------------------------
# Helpers to rebuild docs (SAME STYLE AS TRAINING)
# --------------------------------------------------------------------
def _build_documents(rows: List[CleanVideo]) -> List[str]:
    """
    Build a text document per video for TF-IDF.

    Combines:
      - title_clean or title
      - tags_clean or tags_text
      - description_clean (truncated to 500 chars)
    """
    docs: List[str] = []
    for v in rows:
        title = (v.title_clean or v.title or "").strip()
        tags = (v.tags_clean or v.tags_text or "").strip()
        desc = (v.description_clean or "").strip()

        if len(desc) > 500:
            desc = desc[:500]

        text = f"{title} {tags} {desc}".strip()
        docs.append(text)

    return docs


def _fetch_sample_rows(limit: int = 20000) -> List[CleanVideo]:
    """
    Fetch a subset of CleanVideo rows for visualization.
    Biased towards higher-view videos (as in training).
    """
    q = (
        db.session.query(CleanVideo)
        .filter(CleanVideo.title_clean.isnot(None))
        .order_by(CleanVideo.view_count.desc().nullslast())
        .limit(limit)
    )
    return q.all()


# --------------------------------------------------------------------
# Plot functions
# --------------------------------------------------------------------
def _plot_top_terms_for_cluster(
    cluster_id: int,
    kmeans,
    feature_names: np.ndarray,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Horizontal bar chart of top TF-IDF terms for a given cluster.
    """
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = np.argsort(center)[-top_n:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    top_weights = center[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_terms)), top_weights)
    ax.set_yticks(range(len(top_terms)))
    ax.set_yticklabels(top_terms)
    ax.invert_yaxis()  # highest at top
    ax.set_xlabel("TF-IDF Weight")
    ax.set_title(f"Top {top_n} Terms for Cluster {cluster_id}")

    fig.tight_layout()
    out_path = output_dir / f"cluster_{cluster_id}_top_terms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved top-terms chart to {out_path}")


def _plot_genre_distribution_for_cluster(
    cluster_id: int,
    cluster_genre_counts: Dict[int, Counter],
    output_dir: Path,
) -> None:
    """
    Simple bar chart of genre frequencies inside a chosen cluster.
    """
    genre_counts = cluster_genre_counts.get(cluster_id, Counter())
    if not genre_counts:
        print(f"[WARN] No genre data for cluster {cluster_id}; skipping genre plot.")
        return

    genres = list(genre_counts.keys())
    counts = [genre_counts[g] for g in genres]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(genres, counts)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Videos")
    ax.set_title(f"Genre Distribution in Cluster {cluster_id}")
    ax.set_xticklabels(genres, rotation=45, ha="right")

    fig.tight_layout()
    out_path = output_dir / f"cluster_{cluster_id}_genre_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved genre-distribution chart to {out_path}")


# --------------------------------------------------------------------
# Main visualization pipeline
# --------------------------------------------------------------------
def main():
    app = create_app()
    with app.app_context():
        models_dir = Path("app/ml/models")
        model_path = models_dir / "trend_topic_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find {model_path}. "
                "Run `python -m scripts.train_trend_topic_model` first."
            )

        payload = joblib.load(model_path)
        vectorizer = payload["vectorizer"]
        kmeans = payload["kmeans"]
        cluster_meta = payload["cluster_meta"]

        feature_names = vectorizer.get_feature_names_out()
        n_clusters = kmeans.n_clusters

        print(f"Loaded topic model with {n_clusters} clusters.")
        print("Cluster IDs and dominant genres:")
        for cid, meta in cluster_meta.items():
            print(f"  Cluster {cid}: dominant_genre = {meta.get('dominant_genre', 'Unknown')}")

        # Fetch a subset of videos and predict cluster labels
        print("Fetching sample CleanVideo rows for visualization...")
        rows = _fetch_sample_rows(limit=20000)
        if not rows:
            raise RuntimeError("No CleanVideo rows found for visualization.")

        docs = _build_documents(rows)
        X = vectorizer.transform(docs)
        labels = kmeans.predict(X)

        # Build genre counts per cluster
        cluster_genre_counts: Dict[int, Counter] = defaultdict(Counter)
        for v, label in zip(rows, labels):
            if v.genre:
                cluster_genre_counts[label][v.genre] += 1

        # Choose a cluster to visualize:
        # try to find one with dominant_genre == "Tech"; otherwise cluster 0
        target_cluster = None
        for cid, meta in cluster_meta.items():
            if meta.get("dominant_genre") == "Gaming":
                target_cluster = cid
                break
        if target_cluster is None:
            target_cluster = 0

        print(f"\nSelected cluster {target_cluster} for detailed visualization.")
        print("Dominant genre:",
              cluster_meta[target_cluster].get("dominant_genre", "Unknown"))
        print("Example top_terms:",
              cluster_meta[target_cluster].get("top_terms", [])[:10])

        # Output directory
        output_dir = Path("scripts/plots_topic_model")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Top terms (by TF-IDF weight)
        _plot_top_terms_for_cluster(
            cluster_id=target_cluster,
            kmeans=kmeans,
            feature_names=feature_names,
            output_dir=output_dir,
            top_n=15,
        )

        # 2) Genre distribution in that cluster
        _plot_genre_distribution_for_cluster(
            cluster_id=target_cluster,
            cluster_genre_counts=cluster_genre_counts,
            output_dir=output_dir,
        )

        print("\nDone. Use the generated PNGs in your presentation.")


if __name__ == "__main__":
    main()
