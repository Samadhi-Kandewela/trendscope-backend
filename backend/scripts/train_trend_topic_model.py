"""
Train a topic model on cleaned videos (videos_clean -> CleanVideo).

This model discovers clusters of related topics using TF-IDF + KMeans and
stores, per cluster:
  - dominant_genre       (e.g., "Tech")
  - top_terms            (keywords for the cluster)
  - sample_titles        (representative titles, high views)
  - sample_descriptions  (representative descriptions)

Usage:
    python -m scripts.train_trend_topic_model
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import numpy as np
import joblib

# --------------------------- helpers ---------------------------------

# Extra stopwords to avoid nonsense top_terms like "of", "for", "you", "is", "my"
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union(
    {
        "of", "for", "you", "your", "is", "my", "this", "that", "with",
        "and", "the", "a", "an",
        "official", "video", "channel", "new",
        "2020", "2021", "2022", "2023", "2024", "2025",
        "feat", "ft", "vs", "ep", "episode",
        "shorts", "tiktok", "instagram", "youtube",
    }
)
# sklearn wants list/None/str, not a frozenset
CUSTOM_STOPWORDS_LIST = list(CUSTOM_STOPWORDS)


def _build_training_rows(limit: int | None = None) -> List[CleanVideo]:
    """
    Fetch rows from videos_clean (CleanVideo) for training.

    We bias slightly towards higher-view videos so clusters reflect
    successful content more.
    """
    q = (
        db.session.query(CleanVideo)
        .filter(CleanVideo.title_clean.isnot(None))
        .order_by(CleanVideo.view_count.desc().nullslast())
    )

    if limit:
        q = q.limit(limit)

    rows = q.all()
    return rows


def _build_documents(rows: List[CleanVideo]) -> List[str]:
    """
    Build a text document per video for TF-IDF.

    We combine:
      - title_clean
      - tags_clean / tags_text
      - description_clean (truncated a bit)
    """
    docs: List[str] = []
    for v in rows:
        title = (v.title_clean or v.title or "").strip()
        tags = (v.tags_clean or v.tags_text or "").strip()
        desc = (v.description_clean or "").strip()

        # Optional: truncate very long descriptions to avoid giant docs
        if len(desc) > 500:
            desc = desc[:500]

        text = f"{title} {tags} {desc}".strip()
        docs.append(text)

    return docs


# --------------------------- main training ----------------------------


def train_topic_model():
    app = create_app()
    with app.app_context():
        app.logger.info("Fetching cleaned videos for topic model training...")
        rows = _build_training_rows(limit=120000)  # adjust if needed

        if not rows:
            raise RuntimeError("No rows in videos_clean to train trend topic model.")

        docs = _build_documents(rows)
        app.logger.info("Prepared %d documents for TF-IDF", len(docs))

        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=30000,
            min_df=5,
            max_df=0.5,
            ngram_range=(1, 2),
            stop_words=CUSTOM_STOPWORDS_LIST,
        )
        X = vectorizer.fit_transform(docs)
        app.logger.info("TF-IDF matrix shape: %s", X.shape)

        # Choose a reasonable number of clusters based on data size
        n_docs = X.shape[0]
        if n_docs < 2000:
            n_clusters = 10
        elif n_docs < 20000:
            n_clusters = 20
        else:
            n_clusters = 30

        app.logger.info("Training KMeans with %d clusters...", n_clusters)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)

        feature_names = vectorizer.get_feature_names_out()

        # Build cluster metadata
        cluster_meta: Dict[int, Dict[str, object]] = {}
        cluster_indices: Dict[int, List[int]] = defaultdict(list)

        for idx, label in enumerate(labels):
            cluster_indices[label].append(idx)

        for cluster_id, idxs in cluster_indices.items():
            cluster_videos = [rows[i] for i in idxs]

            # Dominant genre
            genre_counts = Counter(v.genre for v in cluster_videos if v.genre)
            dominant_genre = (
                genre_counts.most_common(1)[0][0]
                if genre_counts
                else "Mixed"
            )

            # Top terms from cluster center
            center = kmeans.cluster_centers_[cluster_id]
            top_n = 15
            top_term_indices = np.argsort(center)[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]

            # Representative titles/descriptions: highest view_count
            sorted_by_views = sorted(
                cluster_videos,
                key=lambda v: v.view_count or 0,
                reverse=True,
            )

            sample_titles = [
                (v.title_clean or v.title or "").strip()
                for v in sorted_by_views[:10]
            ]
            sample_descriptions = [
                (v.description_clean or "").strip()
                for v in sorted_by_views[:10]
            ]

            cluster_meta[cluster_id] = {
                "dominant_genre": dominant_genre,
                "top_terms": top_terms,
                "sample_titles": sample_titles,
                "sample_descriptions": sample_descriptions,
            }

        # Save the whole payload
        models_dir = Path("app/ml/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "trend_topic_model.pkl"

        payload = {
            "vectorizer": vectorizer,
            "kmeans": kmeans,
            "cluster_meta": cluster_meta,
        }

        joblib.dump(payload, out_path)
        app.logger.info("Saved trend topic model to %s", out_path)


if __name__ == "__main__":
    train_topic_model()
