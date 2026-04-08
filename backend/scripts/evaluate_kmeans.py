"""
Phase 2: K-Means Evaluation & Optimisation
===========================================
Evaluates current K-Means clustering model, finds optimal K via Elbow Method,
computes validation metrics, performs coherence inspection, and trains an
optimised model with the best K.

Saves all charts, metrics, and reports to:
  E:/TrendScope/backend/results/phase2_kmeans/

Usage:
    cd E:/TrendScope/backend
    python scripts/evaluate_kmeans.py
    python scripts/evaluate_kmeans.py --sample 5000 --max-k 80
"""

import sys
import os
import json
import time
import shutil
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
import matplotlib.gridspec as gridspec

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo
from app.ml.inference import MODELS_DIR

# ── Results Directory ──────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase2_kmeans"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Same stopwords as original training script ─────────────────────────
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union({
    "of", "for", "you", "your", "is", "my", "this", "that", "with",
    "and", "the", "a", "an",
    "official", "video", "channel", "new",
    "2020", "2021", "2022", "2023", "2024", "2025",
    "feat", "ft", "vs", "ep", "episode",
    "shorts", "tiktok", "instagram", "youtube",
})
CUSTOM_STOPWORDS_LIST = list(CUSTOM_STOPWORDS)


# ═══════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_data(limit: int = 10000) -> List[CleanVideo]:
    print(f"  Loading up to {limit:,} videos from database...")
    videos = (
        db.session.query(CleanVideo)
        .filter(CleanVideo.title_clean.isnot(None), CleanVideo.view_count.isnot(None))
        .order_by(CleanVideo.view_count.desc().nullslast())
        .limit(limit)
        .all()
    )
    print(f"  Loaded {len(videos):,} videos.")
    return videos


def build_texts(videos: List[CleanVideo]) -> List[str]:
    docs = []
    for v in videos:
        title = (v.title_clean or v.title or "").strip()
        tags  = (v.tags_clean  or v.tags_text or "").strip()
        desc  = (v.description_clean or "")[:500].strip()
        docs.append(f"{title} {tags} {desc}".strip())
    return docs


def load_current_model() -> Dict:
    path = MODELS_DIR / "trend_topic_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════
# ELBOW METHOD
# ═══════════════════════════════════════════════════════════════════════

def run_elbow_method(X, k_values: List[int]) -> List[float]:
    print(f"\n  Running Elbow Method: K = {k_values[0]} to {k_values[-1]}...")
    inertias = []
    for k in k_values:
        print(f"    K={k:3d}...", end=" ", flush=True)
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"inertia={km.inertia_:>12,.0f}  ({time.time()-t0:.1f}s)")
    return inertias


def find_optimal_k(k_values: List[int], inertias: List[float]) -> int:
    """Find elbow via maximum second-derivative (rate-of-change)."""
    if len(inertias) < 3:
        return k_values[len(k_values) // 2]
    rates = [
        (inertias[i-1] - inertias[i]) - (inertias[i] - inertias[i+1])
        for i in range(1, len(inertias) - 1)
    ]
    elbow_idx = rates.index(max(rates)) + 1
    return k_values[elbow_idx]


# ═══════════════════════════════════════════════════════════════════════
# CLUSTERING METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(X, labels, sample_size: int = 3000) -> Dict:
    """Compute Silhouette, Davies-Bouldin, Calinski-Harabasz."""
    n_unique = len(set(labels))
    if n_unique < 2:
        return {'silhouette': 0.0, 'davies_bouldin': 999.0, 'calinski_harabasz': 0.0}

    n = X.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        X_s = X[idx]
        y_s = np.array(labels)[idx]
    else:
        X_s = X
        y_s = np.array(labels)

    X_dense = X_s.toarray() if hasattr(X_s, 'toarray') else X_s

    sil = silhouette_score(X_dense, y_s, sample_size=min(2000, len(y_s)))
    db  = davies_bouldin_score(X_dense, y_s)
    ch  = calinski_harabasz_score(X_dense, y_s)

    return {
        'silhouette':        round(float(sil), 4),
        'davies_bouldin':    round(float(db),  4),
        'calinski_harabasz': round(float(ch),  2),
    }


def compute_silhouette_across_k(X, k_values_subset: List[int]) -> List[float]:
    """Compute silhouette score for each K value (for chart)."""
    scores = []
    for k in k_values_subset:
        print(f"    Silhouette K={k:3d}...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
        labels = km.fit_predict(X)
        m = compute_metrics(X, labels, sample_size=2000)
        scores.append(m['silhouette'])
        print(f"{m['silhouette']:.4f}")
    return scores


# ═══════════════════════════════════════════════════════════════════════
# COHERENCE INSPECTION
# ═══════════════════════════════════════════════════════════════════════

def inspect_coherence(videos, labels, n_top: int = 20):
    """
    Inspect top N clusters by size.
    Labels each as Coherent / Partially Coherent / Incoherent.
    """
    import re
    from app.ml.niche_filter import NICHE_TAXONOMY
    from app.utils.text_utils import extract_top_keywords

    all_niche_words = set()
    for vocab in NICHE_TAXONOMY.values():
        all_niche_words.update(vocab)

    cluster_videos = defaultdict(list)
    for v, lbl in zip(videos, labels):
        cluster_videos[int(lbl)].append(v)

    sorted_clusters = sorted(cluster_videos.items(), key=lambda x: len(x[1]), reverse=True)

    year_re = re.compile(r'^20\d{2}$')
    report = []
    coherent = partial = incoherent = 0

    for cid, vids in sorted_clusters[:n_top]:
        texts = [
            f"{v.title_clean or v.title or ''} {v.tags_clean or v.tags_text or ''}"
            for v in vids
        ]
        kws = [k for k in extract_top_keywords(texts, top_n=12) if not year_re.match(k)][:10]

        genre_counts = Counter(v.genre for v in vids if v.genre)
        dominant = genre_counts.most_common(1)[0][0] if genre_counts else "Unknown"

        niche_hit = sum(1 for kw in kws if kw.lower() in all_niche_words)
        ratio = niche_hit / max(len(kws), 1)

        if ratio >= 0.7:
            label = "Coherent";          coherent  += 1
        elif ratio >= 0.4:
            label = "Partially Coherent"; partial   += 1
        else:
            label = "Incoherent";          incoherent += 1

        report.append({
            'cluster_id':    cid,
            'size':          len(vids),
            'dominant_genre': dominant,
            'top_keywords':  kws,
            'niche_pct':     round(ratio * 100, 1),
            'label':         label,
        })

    return report, coherent, partial, incoherent


# ═══════════════════════════════════════════════════════════════════════
# RETRAIN OPTIMISED MODEL
# ═══════════════════════════════════════════════════════════════════════

def retrain_optimised(videos: List[CleanVideo], optimal_k: int) -> Tuple[np.ndarray, Dict]:
    """
    Retrain K-Means with optimal_k using the same methodology as the
    original train_trend_topic_model.py script.
    """
    print(f"\n  Retraining with optimal K={optimal_k}...")

    docs = build_texts(videos)

    vec = TfidfVectorizer(
        max_features=30000,
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 2),
        stop_words=CUSTOM_STOPWORDS_LIST,
    )
    X_new = vec.fit_transform(docs)
    print(f"  TF-IDF matrix shape: {X_new.shape}")

    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_new)

    feature_names = vec.get_feature_names_out()
    cluster_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_indices[int(lbl)].append(idx)

    cluster_meta: Dict[int, Dict] = {}
    for cid, idxs in cluster_indices.items():
        cluster_vids = [videos[i] for i in idxs]

        genre_counts = Counter(v.genre for v in cluster_vids if v.genre)
        dominant = genre_counts.most_common(1)[0][0] if genre_counts else "Mixed"

        center = km.cluster_centers_[cid]
        top_term_idxs = np.argsort(center)[-15:][::-1]
        top_terms = [feature_names[i] for i in top_term_idxs]

        sorted_vids = sorted(cluster_vids, key=lambda v: v.view_count or 0, reverse=True)
        cluster_meta[cid] = {
            "dominant_genre":      dominant,
            "top_terms":           top_terms,
            "sample_titles":       [(v.title_clean or v.title or "").strip() for v in sorted_vids[:10]],
            "sample_descriptions": [(v.description_clean or "").strip() for v in sorted_vids[:10]],
            "size":                len(cluster_vids),
        }

    payload = {
        "vectorizer":   vec,
        "kmeans":       km,
        "cluster_meta": cluster_meta,
        "k":            optimal_k,
        "trained_at":   datetime.utcnow().isoformat(),
        "evaluation":   "elbow-validated",
    }

    # Backup original, save optimised separately
    original = MODELS_DIR / "trend_topic_model.pkl"
    backup   = MODELS_DIR / "trend_topic_model_backup.pkl"
    optimised = MODELS_DIR / "trend_topic_model_optimised.pkl"

    if original.exists() and not backup.exists():
        shutil.copy2(original, backup)
        print(f"  Backed up original -> {backup.name}")

    joblib.dump(payload, optimised)
    print(f"  Saved optimised model -> {optimised.name}")

    return labels, X_new, payload


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

def chart_elbow(k_values, inertias, optimal_k):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, inertias, 'bo-', lw=2, ms=8)
    ax.axvline(x=optimal_k, color='red',    ls='--', lw=2, label=f'Elbow / Optimal K={optimal_k}')
    ax.axvline(x=50,        color='orange', ls=':',  lw=2, label='Current K=50')
    for k, v in zip(k_values, inertias):
        ax.annotate(f'{v:,.0f}', (k, v), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, color='navy')
    ax.set_xlabel('Number of Clusters (K)', fontsize=13)
    ax.set_ylabel('Inertia (WCSS)', fontsize=13)
    ax.set_title('K-Means Elbow Method\nFinding Optimal Number of Clusters', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xticks(k_values)
    plt.tight_layout()
    _save(fig, "01_elbow_curve.png")


def chart_silhouette_vs_k(k_vals, scores, optimal_k):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_vals, scores, 'go-', lw=2, ms=10)
    best_i = scores.index(max(scores))
    ax.annotate(f'Best: {max(scores):.4f}\nat K={k_vals[best_i]}',
                xy=(k_vals[best_i], max(scores)),
                xytext=(k_vals[best_i] + 2, max(scores) + 0.005),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.axvline(x=50,        color='orange', ls=':',  lw=2, label='Current K=50')
    ax.axvline(x=optimal_k, color='red',    ls='--', lw=2, label=f'Elbow K={optimal_k}')
    ax.set_xlabel('K', fontsize=13); ax.set_ylabel('Silhouette Score', fontsize=13)
    ax.set_title('Silhouette Score vs K\nHigher = Better Cluster Quality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "02_silhouette_vs_k.png")


def chart_metrics_comparison(bm, om, k_base, k_opt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    specs = [
        ('silhouette',        'Silhouette Score',      'Higher is better',  True),
        ('davies_bouldin',    'Davies-Bouldin Index',  'Lower is better',   False),
        ('calinski_harabasz', 'Calinski-Harabasz',     'Higher is better',  True),
    ]
    for ax, (key, title, sub, hi) in zip(axes, specs):
        bv, ov = bm[key], om[key]
        colors = ['#2196F3', '#4CAF50'] if hi else ['#2196F3', '#FF9800']
        bars = ax.bar([f'K={k_base}\nBaseline', f'K={k_opt}\nOptimised'],
                      [bv, ov], color=colors, width=0.5, edgecolor='black')
        for bar, val in zip(bars, [bv, ov]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                    f'{val:.4f}', ha='center', fontweight='bold', fontsize=12)
        imp = ((ov - bv) / max(abs(bv), 1e-6) * 100) if hi else ((bv - ov) / max(abs(bv), 1e-6) * 100)
        col = 'green' if imp > 0 else 'red'
        ax.text(0.5, 0.95, f'{"+" if imp > 0 else ""}{imp:.1f}%',
                transform=ax.transAxes, ha='center', va='top',
                color=col, fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=col))
        ax.set_title(f'{title}\n{sub}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f'Clustering Quality: K={k_base} vs K={k_opt}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "03_metrics_comparison.png")


def chart_cluster_sizes(labels_base, labels_opt, k_base, k_opt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, labels, k, title in [
        (ax1, labels_base, k_base, f'K={k_base} (Baseline)'),
        (ax2, labels_opt,  k_opt,  f'K={k_opt} (Optimised)'),
    ]:
        sizes = sorted(Counter(labels).values(), reverse=True)
        ax.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7, edgecolor='black', lw=0.5)
        ax.axhline(np.mean(sizes),   color='red',   ls='--', label=f'Mean {np.mean(sizes):.0f}')
        ax.axhline(np.median(sizes), color='green', ls=':',  label=f'Median {np.median(sizes):.0f}')
        ax.set_title(f'Cluster Size Distribution\n{title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Clusters (sorted by size)'); ax.set_ylabel('Videos')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        ax.text(0.98, 0.97,
                f'Min: {min(sizes)}\nMax: {max(sizes)}\nStd: {np.std(sizes):.0f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    _save(fig, "04_cluster_size_distribution.png")


def chart_coherence(coherent, partial, incoherent, k, total):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    labels = ['Coherent', 'Partially\nCoherent', 'Incoherent']
    sizes  = [coherent, partial, incoherent]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=(0.05, 0.05, 0.05), shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f'Cluster Coherence\n(Top {total} clusters, K={k})', fontsize=13, fontweight='bold')
    bars = ax2.bar(labels, sizes, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 str(val), ha='center', fontweight='bold', fontsize=13)
    ax2.set_title(f'Coherence Count (out of {total})', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Clusters'); ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(sizes) * 1.25)
    plt.tight_layout()
    _save(fig, "05_coherence_distribution.png")


def _save(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ═══════════════════════════════════════════════════════════════════════

def write_coherence_report(report, path):
    icons = {'Coherent': '[OK]', 'Partially Coherent': '[~~]', 'Incoherent': '[XX]'}
    with open(path, 'w', encoding='utf-8') as f:
        f.write("CLUSTER COHERENCE INSPECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        for item in report:
            icon = icons.get(item['label'], '[??]')
            f.write(
                f"{icon} Cluster {item['cluster_id']:3d} | "
                f"Size: {item['size']:5d} | "
                f"Genre: {item['dominant_genre']:15s} | "
                f"{item['label']} ({item['niche_pct']:.0f}%)\n"
            )
            f.write(f"     Keywords: {', '.join(item['top_keywords'])}\n\n")


def write_summary(results: Dict, path):
    bm = results['baseline_metrics']
    om = results['optimised_metrics']
    ok = results['optimal_k']
    sil_imp = (om['silhouette'] - bm['silhouette']) / max(abs(bm['silhouette']), 1e-6) * 100
    db_imp  = (bm['davies_bouldin'] - om['davies_bouldin']) / max(abs(bm['davies_bouldin']), 1e-6) * 100
    ch_imp  = (om['calinski_harabasz'] - bm['calinski_harabasz']) / max(abs(bm['calinski_harabasz']), 1e-6) * 100
    total   = results['coherent'] + results['partial'] + results['incoherent']

    with open(path, 'w', encoding='utf-8') as f:
        f.write("PHASE 2: K-MEANS EVALUATION — EXECUTIVE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Dataset   : {results['n_videos']:,} videos\n\n")

        f.write("1. CURRENT MODEL (K=50)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Silhouette Score     : {bm['silhouette']:.4f}\n")
        f.write(f"  Davies-Bouldin Index : {bm['davies_bouldin']:.4f}\n")
        f.write(f"  Calinski-Harabasz    : {bm['calinski_harabasz']:.2f}\n")
        interp = 'poor' if bm['silhouette'] < 0.10 else ('weak' if bm['silhouette'] < 0.25 else 'moderate')
        f.write(f"  Interpretation       : {interp} cluster structure\n\n")

        f.write("2. ELBOW METHOD\n")
        f.write("-" * 40 + "\n")
        f.write(f"  K values tested : {results['k_values']}\n")
        f.write(f"  Optimal K found : {ok}\n\n")

        f.write(f"3. OPTIMISED MODEL (K={ok})\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Silhouette Score     : {om['silhouette']:.4f}  ({sil_imp:+.1f}%)\n")
        f.write(f"  Davies-Bouldin Index : {om['davies_bouldin']:.4f}  ({db_imp:+.1f}%)\n")
        f.write(f"  Calinski-Harabasz    : {om['calinski_harabasz']:.2f}   ({ch_imp:+.1f}%)\n\n")

        f.write("4. COHERENCE INSPECTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Coherent           : {results['coherent']}/{total}  ({results['coherent']/max(total,1)*100:.0f}%)\n")
        f.write(f"  Partially Coherent : {results['partial']}/{total}\n")
        f.write(f"  Incoherent         : {results['incoherent']}/{total}\n\n")

        f.write("5. K-MEANS KNOWN LIMITATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("  a) Hard assignment: every video is forced into a cluster\n"
                "     even if it is off-niche noise.\n")
        f.write("  b) Fixed K: assumes all clusters have similar size/density.\n"
                "     YouTube genres have very unequal distributions.\n")
        f.write("  c) TF-IDF semantic gap: 'game review' and 'game makeup'\n"
                "     appear similar because TF-IDF only counts word frequency.\n")
        f.write("  d) Confidence formula: softmax over 50 distances is\n"
                "     mathematically bounded at ~0.20-0.25 (addressed in Phase 5).\n\n")

        f.write("6. CONCLUSIONS & NEXT STEPS\n")
        f.write("-" * 40 + "\n")
        if ok != 50:
            f.write(f"  -> K={ok} is the Elbow-validated optimum (vs arbitrary K=50)\n")
        f.write(f"  -> Proceed to Phase 3 (LDA) for better keyword quality\n")
        f.write(f"  -> Proceed to Phase 4 (Niche Filter fixes)\n")
        f.write(f"  -> Proceed to Phase 5 (Composite Confidence Score)\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(sample_size: int = 10000, max_k: int = 80, k_step: int = 10):
    app = create_app()
    with app.app_context():
        print("\n" + "=" * 60)
        print("  PHASE 2: K-MEANS EVALUATION & OPTIMISATION")
        print("=" * 60)
        print(f"  Results dir : {RESULTS_DIR}")
        print(f"  Sample size : {sample_size:,}")
        print(f"  K range     : 10 to {max_k} (step {k_step})")
        print("=" * 60)

        results: Dict = {'generatedAt': datetime.now().isoformat()}

        # ── 1. Load data ───────────────────────────────────────────────
        print("\n[1/8] Loading data...")
        videos = load_data(sample_size)
        results['n_videos'] = len(videos)

        # ── 2. Load existing model & transform data ────────────────────
        print("\n[2/8] Loading existing model and transforming data...")
        payload_current = load_current_model()
        vec_current  = payload_current['vectorizer']
        km_current   = payload_current['kmeans']
        k_current    = km_current.n_clusters
        print(f"  Current model K = {k_current}")

        texts = build_texts(videos)
        X = vec_current.transform(texts)
        print(f"  Feature matrix : {X.shape}")
        results['features'] = X.shape[1]

        # ── 3. Elbow Method ────────────────────────────────────────────
        print("\n[3/8] Running Elbow Method...")
        k_values = sorted(set(list(range(10, max_k + 1, k_step)) + [k_current]))
        inertias = run_elbow_method(X, k_values)
        optimal_k = find_optimal_k(k_values, inertias)
        results['k_values'] = k_values
        results['inertias'] = inertias
        results['optimal_k'] = optimal_k
        print(f"\n  Elbow identified at: K={optimal_k}")
        chart_elbow(k_values, inertias, optimal_k)

        # ── 4. Metrics for current model ───────────────────────────────
        print(f"\n[4/8] Computing metrics for current model (K={k_current})...")
        labels_current = km_current.predict(X).tolist()
        bm = compute_metrics(X, labels_current)
        results['baseline_metrics'] = bm
        print(f"  Silhouette     : {bm['silhouette']:.4f}")
        print(f"  Davies-Bouldin : {bm['davies_bouldin']:.4f}")
        print(f"  Calinski-Harab : {bm['calinski_harabasz']:.2f}")

        # ── 5. Silhouette across K ──────────────────────────────────────
        print("\n[5/8] Silhouette scores across K range...")
        k_sub = k_values[::2]
        sil_scores = compute_silhouette_across_k(X, k_sub)
        results['silhouette_k_values'] = k_sub
        results['silhouette_scores']   = sil_scores
        chart_silhouette_vs_k(k_sub, sil_scores, optimal_k)

        # ── 6. Retrain with optimal K ──────────────────────────────────
        print(f"\n[6/8] Retraining with optimal K={optimal_k}...")
        labels_opt, X_opt, payload_opt = retrain_optimised(videos, optimal_k)
        om = compute_metrics(X_opt, labels_opt)
        results['optimised_metrics'] = om
        print(f"  Silhouette     : {om['silhouette']:.4f}")
        print(f"  Davies-Bouldin : {om['davies_bouldin']:.4f}")
        print(f"  Calinski-Harab : {om['calinski_harabasz']:.2f}")

        # ── 7. Coherence Inspection ────────────────────────────────────
        print(f"\n[7/8] Coherence inspection (top {min(20, optimal_k)} clusters)...")
        n_inspect = min(20, optimal_k)
        coh_report, coherent, partial, incoherent = inspect_coherence(
            videos, labels_opt, n_top=n_inspect
        )
        results['coherent']   = coherent
        results['partial']    = partial
        results['incoherent'] = incoherent
        total = coherent + partial + incoherent
        print(f"  Coherent            : {coherent}/{total}  ({coherent/max(total,1)*100:.0f}%)")
        print(f"  Partially Coherent  : {partial}/{total}")
        print(f"  Incoherent          : {incoherent}/{total}")

        # ── 8. Save all outputs ────────────────────────────────────────
        print("\n[8/8] Saving charts and reports...")
        chart_metrics_comparison(bm, om, k_current, optimal_k)
        chart_cluster_sizes(labels_current, labels_opt, k_current, optimal_k)
        chart_coherence(coherent, partial, incoherent, optimal_k, total)

        (RESULTS_DIR / "evaluation_results.json").write_text(
            json.dumps(results, indent=2), encoding='utf-8'
        )
        print(f"  Saved: evaluation_results.json")

        write_coherence_report(coh_report, RESULTS_DIR / "cluster_coherence_report.txt")
        print(f"  Saved: cluster_coherence_report.txt")

        write_summary(results, RESULTS_DIR / "phase2_summary.txt")
        print(f"  Saved: phase2_summary.txt")

        # ── Final console summary ──────────────────────────────────────
        sil_imp = (om['silhouette'] - bm['silhouette']) / max(abs(bm['silhouette']), 1e-6) * 100
        db_imp  = (bm['davies_bouldin'] - om['davies_bouldin']) / max(abs(bm['davies_bouldin']), 1e-6) * 100

        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        print(f"  {'Metric':<22} {'K='+str(k_current)+' Baseline':>14}  {'K='+str(optimal_k)+' Optimised':>14}  Change")
        print(f"  {'-'*58}")
        print(f"  {'Silhouette Score':<22} {bm['silhouette']:>14.4f}  {om['silhouette']:>14.4f}  {sil_imp:+.1f}%")
        print(f"  {'Davies-Bouldin':<22} {bm['davies_bouldin']:>14.4f}  {om['davies_bouldin']:>14.4f}  {db_imp:+.1f}%")
        print(f"  {'Calinski-Harabasz':<22} {bm['calinski_harabasz']:>14.2f}  {om['calinski_harabasz']:>14.2f}")
        print(f"\n  Optimal K (Elbow): {optimal_k}")
        print(f"  Coherence Rate   : {coherent}/{total} ({coherent/max(total,1)*100:.0f}%)")
        print(f"\n  All results saved to:")
        print(f"  {RESULTS_DIR}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: K-Means Evaluation")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-k",  type=int, default=80)
    parser.add_argument("--k-step", type=int, default=10)
    args = parser.parse_args()
    main(sample_size=args.sample, max_k=args.max_k, k_step=args.k_step)
