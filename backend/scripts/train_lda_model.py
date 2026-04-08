"""
Phase 2C: LDA Topic Model (Alternative to K-Means Clustering)
==============================================================
LDA (Latent Dirichlet Allocation) is purpose-built for text topic discovery.
Unlike K-Means, LDA:
  - Does NOT require choosing K in advance (but we set n_topics)
  - Models each document as a mixture of topics (soft assignment)
  - Uses word co-occurrence patterns, not just distance in vector space
  - Naturally produces human-readable topics

This script:
  1. Loads the same English-filtered data as retrain_with_svd.py
  2. Trains LDA with 20 topics
  3. Evaluates coherence and compares against SVD+K-Means
  4. Saves the model for use as an alternative in trend_service.py

Saves: app/ml/models/trend_lda_model.pkl
Results: results/phase2_lda/

Usage:
    cd E:/TrendScope/backend
    python scripts/train_lda_model.py
    python scripts/train_lda_model.py --topics 25 --sample 15000
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo
from app.ml.inference import MODELS_DIR
from app.utils.text_utils import extract_top_keywords

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase2_lda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS.union({
    "of", "for", "you", "your", "is", "my", "this", "that", "with",
    "and", "the", "a", "an", "official", "video", "channel", "new",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    "feat", "ft", "vs", "ep", "episode", "shorts", "tiktok", "youtube",
}))

ENGLISH_REGIONS = {'US', 'GB', 'CA', 'AU', 'NZ', 'IE'}


# ═══════════════════════════════════════════════════════════════════════
# LANGUAGE FILTER (same as retrain_with_svd.py)
# ═══════════════════════════════════════════════════════════════════════

def is_english_dominant(text: str, threshold: float = 0.75) -> bool:
    if not text or len(text) < 3:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold


def filter_english_videos(videos: List[CleanVideo]) -> List[CleanVideo]:
    filtered = []
    for v in videos:
        title = (v.title_clean or v.title or "").strip()
        country = (v.trending_country_code or "").upper()
        if is_english_dominant(title, 0.75) and country in ENGLISH_REGIONS:
            filtered.append(v)
    return filtered


# ═══════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_data(limit: int) -> List[CleanVideo]:
    print(f"  Loading up to {limit:,} videos...")
    return (
        db.session.query(CleanVideo)
        .filter(
            CleanVideo.title_clean.isnot(None),
            CleanVideo.view_count.isnot(None),
            CleanVideo.view_count > 0,
            CleanVideo.source_dataset != 'most_popular',
        )
        .order_by(CleanVideo.view_count.desc())
        .limit(limit)
        .all()
    )


def build_texts(videos: List[CleanVideo]) -> List[str]:
    docs = []
    for v in videos:
        title = (v.title_clean or v.title or "").strip()
        tags  = (v.tags_clean or v.tags_text or "").strip()
        desc  = (v.description_clean or "")[:300].strip()
        docs.append(f"{title} {tags} {desc}".strip())
    return docs


# ═══════════════════════════════════════════════════════════════════════
# LDA TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_lda(texts: List[str], n_topics: int) -> Tuple:
    """
    Trains LDA model on text corpus.

    Note: LDA requires CountVectorizer (raw word counts), NOT TF-IDF.
    TF-IDF weights would distort LDA's probabilistic document model.
    """
    print(f"  Building CountVectorizer (LDA requires counts, not TF-IDF)...")
    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=3,
        max_df=0.6,
        ngram_range=(1, 2),
        stop_words=CUSTOM_STOPWORDS,
    )
    X_counts = vectorizer.fit_transform(texts)
    print(f"  Count matrix shape: {X_counts.shape}")

    print(f"  Training LDA with {n_topics} topics (max_iter=20)...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        learning_offset=50.0,
        random_state=42,
        n_jobs=-1,
    )
    lda.fit(X_counts)

    perplexity = lda.perplexity(X_counts)
    print(f"  LDA perplexity: {perplexity:.1f}  (lower = better fit)")

    return vectorizer, lda, X_counts, perplexity


# ═══════════════════════════════════════════════════════════════════════
# CLUSTER METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_lda_topic_terms(lda: LatentDirichletAllocation,
                            vectorizer: CountVectorizer,
                            n_top: int = 12) -> Dict[int, List[str]]:
    """
    Extracts top n words for each LDA topic from the LDA component matrix.
    These are the words with highest probability in each topic.
    """
    feature_names = vectorizer.get_feature_names_out()
    topic_terms = {}
    import re
    year_re = re.compile(r'^20\d{2}$')

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top:][::-1]
        terms = [
            feature_names[i] for i in top_indices
            if not year_re.match(feature_names[i]) and len(feature_names[i]) > 2
        ]
        topic_terms[topic_idx] = terms

    return topic_terms


def build_lda_cluster_meta(videos: List[CleanVideo],
                           labels: List[int],
                           topic_terms: Dict[int, List[str]]) -> Dict:
    """
    Builds cluster_meta compatible with trend_service.py expectations.
    Each entry has: dominant_genre, top_terms, top_keywords, size, sample_titles
    """
    cluster_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_indices[int(lbl)].append(idx)

    cluster_meta = {}
    for cid, idxs in cluster_indices.items():
        cluster_vids = [videos[i] for i in idxs]

        genre_counts = Counter(v.genre for v in cluster_vids if v.genre)
        dominant_genre = genre_counts.most_common(1)[0][0] if genre_counts else "Mixed"

        # Use LDA's probabilistic topic terms (better than frequency extraction)
        terms = topic_terms.get(cid, [])

        sorted_vids = sorted(cluster_vids, key=lambda v: v.view_count or 0, reverse=True)
        sample_titles = [(v.title_clean or v.title or "").strip() for v in sorted_vids[:5]]

        cluster_meta[cid] = {
            "dominant_genre": dominant_genre,
            "top_terms":      terms,
            "top_keywords":   terms,   # alias for compatibility
            "size":           len(cluster_vids),
            "sample_titles":  sample_titles,
        }

    return cluster_meta


# ═══════════════════════════════════════════════════════════════════════
# COHERENCE INSPECTION
# ═══════════════════════════════════════════════════════════════════════

def inspect_coherence(cluster_meta: Dict) -> Tuple[List[Dict], int, int, int]:
    from app.ml.niche_filter import NICHE_TAXONOMY
    all_niche_words = set()
    for vocab in NICHE_TAXONOMY.values():
        all_niche_words.update(vocab)

    sorted_clusters = sorted(cluster_meta.items(), key=lambda x: x[1]["size"], reverse=True)

    report = []
    coherent = partial = incoherent = 0

    for cid, meta in sorted_clusters:
        kws = meta.get("top_terms", [])
        niche_hits = sum(1 for kw in kws if kw.lower() in all_niche_words)
        ratio = niche_hits / max(len(kws), 1)

        if ratio >= 0.7:
            label = "Coherent";           coherent  += 1
        elif ratio >= 0.4:
            label = "Partially Coherent"; partial   += 1
        else:
            label = "Incoherent";         incoherent += 1

        report.append({
            "cluster_id":     cid,
            "size":           meta["size"],
            "dominant_genre": meta["dominant_genre"],
            "top_keywords":   kws,
            "niche_pct":      round(ratio * 100, 1),
            "label":          label,
        })

    return report, coherent, partial, incoherent


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

def save_chart(fig, filename: str):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {path.name}")


def chart_topic_word_distributions(topic_terms: Dict[int, List[str]], n_show: int = 12):
    """Visualise top words for each topic."""
    n_topics = len(topic_terms)
    cols = 4
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if n_topics > 1 else [axes]

    for i, (tid, terms) in enumerate(topic_terms.items()):
        ax = axes[i]
        ax.barh(range(len(terms[:10])), [1] * len(terms[:10]), color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(terms[:10])))
        ax.set_yticklabels(terms[:10], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'Topic {tid}', fontsize=10, fontweight='bold')
        ax.set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('LDA Topic Word Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, "01_lda_topics.png")


def chart_coherence_comparison(svd_result: Dict, lda_coherent: int,
                               lda_partial: int, lda_incoherent: int, n_topics: int):
    """Compare SVD+K-Means vs LDA coherence side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ['Coherent', 'Partially\nCoherent', 'Incoherent']
    colors = ['#4CAF50', '#FF9800', '#F44336']

    # Load SVD results if available
    svd_path = Path(__file__).resolve().parent.parent / "results" / "phase2_svd" / "svd_results.json"
    svd_coherent = svd_partial = svd_incoherent = 0
    if svd_path.exists():
        with open(svd_path) as f:
            svd_data = json.load(f)
        svd_coherent   = svd_data.get('coherent', 0)
        svd_partial    = svd_data.get('partial', 0)
        svd_incoherent = svd_data.get('incoherent', 0)

    for ax, title, sizes in [
        (axes[0], f'SVD + K-Means (K={svd_data.get("optimal_k","?") if svd_path.exists() else "?"})',
         [svd_coherent, svd_partial, svd_incoherent]),
        (axes[1], f'LDA ({n_topics} topics)',
         [lda_coherent, lda_partial, lda_incoherent]),
    ]:
        bars = ax.bar(labels, sizes, color=colors, edgecolor='black', width=0.5)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    str(val), ha='center', fontweight='bold', fontsize=13)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Clusters'); ax.grid(True, alpha=0.3, axis='y')
        total = sum(sizes)
        ax.set_ylim(0, max(sizes) * 1.3 if max(sizes) > 0 else 5)
        coherence_rate = sizes[0] / max(total, 1) * 100
        ax.text(0.5, 0.92, f'Coherence: {coherence_rate:.0f}%',
                transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='green'))

    plt.suptitle('Model Comparison: SVD+K-Means vs LDA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, "02_svd_vs_lda_coherence.png")


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def write_topic_report(report: List[Dict], path: Path):
    icons = {'Coherent': '[OK]', 'Partially Coherent': '[~~]', 'Incoherent': '[XX]'}
    with open(path, 'w', encoding='utf-8') as f:
        f.write("LDA TOPIC COHERENCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        for item in report:
            icon = icons.get(item['label'], '[??]')
            f.write(
                f"{icon} Topic {item['cluster_id']:3d} | "
                f"Size: {item['size']:5d} | "
                f"Genre: {item['dominant_genre']:15s} | "
                f"{item['label']} ({item['niche_pct']:.0f}%)\n"
            )
            f.write(f"     Terms: {', '.join(item['top_keywords'])}\n\n")


def write_summary(results: Dict, path: Path):
    total = results['coherent'] + results['partial'] + results['incoherent']
    with open(path, 'w', encoding='utf-8') as f:
        f.write("PHASE 2C: LDA MODEL — SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Topics     : {results['n_topics']}\n")
        f.write(f"Videos     : {results['filtered_videos']:,} (English-filtered)\n")
        f.write(f"Perplexity : {results['perplexity']:.1f}  (lower = better)\n\n")

        f.write("COHERENCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Coherent           : {results['coherent']}/{total}  ({results['coherent']/max(total,1)*100:.0f}%)\n")
        f.write(f"  Partially Coherent : {results['partial']}/{total}\n")
        f.write(f"  Incoherent         : {results['incoherent']}/{total}\n\n")

        f.write("COMPARISON WITH SVD+K-MEANS\n")
        f.write("-" * 40 + "\n")
        svd_path = Path(__file__).resolve().parent.parent / "results" / "phase2_svd" / "svd_results.json"
        if svd_path.exists():
            with open(svd_path) as svd_f:
                svd_data = json.load(svd_f)
            svd_coh = svd_data.get('coherent', 0)
            svd_tot = svd_coh + svd_data.get('partial', 0) + svd_data.get('incoherent', 0)
            f.write(f"  SVD+K-Means coherence : {svd_coh}/{svd_tot} ({svd_coh/max(svd_tot,1)*100:.0f}%)\n")
            f.write(f"  LDA coherence         : {results['coherent']}/{total} ({results['coherent']/max(total,1)*100:.0f}%)\n")
            winner = "LDA" if results['coherent'] > svd_coh else "SVD+K-Means"
            f.write(f"  -> Better model       : {winner}\n")
        f.write("\nBoth models saved. trend_service.py supports modelType='lda' and 'svd'.\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(sample_size: int = 15000, n_topics: int = 20):
    app = create_app()
    with app.app_context():
        print("\n" + "=" * 60)
        print("  PHASE 2C: LDA TOPIC MODEL TRAINING")
        print("=" * 60)
        print(f"  Topics      : {n_topics}")
        print(f"  Sample size : {sample_size:,}")
        print(f"  Results dir : {RESULTS_DIR}")
        print("=" * 60)

        results = {"generatedAt": datetime.now().isoformat(), "n_topics": n_topics}

        # ── 1. Load & Filter ───────────────────────────────────────────
        print("\n[1/6] Loading and filtering data...")
        from app.models.clean_video import CleanVideo
        videos_all = (
            db.session.query(CleanVideo)
            .filter(
                CleanVideo.title_clean.isnot(None),
                CleanVideo.view_count > 0,
                CleanVideo.source_dataset != 'most_popular',
            )
            .order_by(CleanVideo.view_count.desc())
            .limit(sample_size)
            .all()
        )
        videos = filter_english_videos(videos_all)
        results['total_videos']    = len(videos_all)
        results['filtered_videos'] = len(videos)
        print(f"  Total loaded  : {len(videos_all):,}")
        print(f"  After filter  : {len(videos):,} English videos")

        if len(videos) < 200:
            print("ERROR: Not enough English videos. Exiting.")
            return

        texts = build_texts(videos)

        # ── 2. Train LDA ───────────────────────────────────────────────
        print("\n[2/6] Training LDA model...")
        vectorizer, lda, X_counts, perplexity = train_lda(texts, n_topics)
        results['perplexity'] = round(perplexity, 2)

        # ── 3. Get topic distributions & assign labels ─────────────────
        print("\n[3/6] Assigning documents to topics...")
        X_topics = lda.transform(X_counts)          # (n_docs, n_topics)
        labels = X_topics.argmax(axis=1).tolist()   # dominant topic per doc
        confidences = X_topics.max(axis=1).tolist() # probability of dominant topic
        avg_confidence = round(float(np.mean(confidences)), 3)
        print(f"  Avg topic confidence: {avg_confidence:.3f}")
        print(f"  Topic distribution: {Counter(labels).most_common(5)}")

        # ── 4. Extract topic terms & build metadata ────────────────────
        print("\n[4/6] Extracting topic terms and building cluster metadata...")
        topic_terms = extract_lda_topic_terms(lda, vectorizer, n_top=12)
        cluster_meta = build_lda_cluster_meta(videos, labels, topic_terms)

        chart_topic_word_distributions(topic_terms)

        # ── 5. Coherence inspection ────────────────────────────────────
        print("\n[5/6] Coherence inspection...")
        coh_report, coherent, partial, incoherent = inspect_coherence(cluster_meta)
        total = coherent + partial + incoherent
        results['coherent']   = coherent
        results['partial']    = partial
        results['incoherent'] = incoherent
        print(f"  Coherent           : {coherent}/{total}  ({coherent/max(total,1)*100:.0f}%)")
        print(f"  Partially Coherent : {partial}/{total}")
        print(f"  Incoherent         : {incoherent}/{total}")

        chart_coherence_comparison({}, coherent, partial, incoherent, n_topics)

        # ── 6. Save model ──────────────────────────────────────────────
        print("\n[6/6] Saving LDA model...")
        payload = {
            "vectorizer":   vectorizer,
            "lda":          lda,
            "topic_terms":  topic_terms,
            "cluster_meta": cluster_meta,
            "k":            n_topics,
            "model_type":   "lda",
            "trained_at":   datetime.utcnow().isoformat(),
            "evaluation": {
                "perplexity":        round(perplexity, 2),
                "coherent_clusters": coherent,
                "total_topics":      total,
                "coherence_rate":    round(coherent / max(total, 1), 3),
                "avg_confidence":    avg_confidence,
                "filtered_videos":   len(videos),
            }
        }

        model_path = MODELS_DIR / "trend_lda_model.pkl"
        joblib.dump(payload, model_path)
        print(f"  Model saved : {model_path}")

        (RESULTS_DIR / "lda_results.json").write_text(
            json.dumps(results, indent=2), encoding='utf-8'
        )
        write_topic_report(coh_report, RESULTS_DIR / "lda_topic_report.txt")
        write_summary(results, RESULTS_DIR / "lda_summary.txt")

        print("\n" + "=" * 60)
        print("  LDA RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Topics trained     : {n_topics}")
        print(f"  Perplexity         : {perplexity:.1f}")
        print(f"  Coherent topics    : {coherent}/{total}  ({coherent/max(total,1)*100:.0f}%)")
        print(f"  Avg confidence     : {avg_confidence:.3f}")
        print(f"  Model              : {model_path.name}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--sample", type=int, default=15000)
    args = parser.parse_args()
    main(sample_size=args.sample, n_topics=args.topics)
