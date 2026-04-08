"""
Training Script: Trend Classifier
==================================
Run this script to train the supervised trend classification model.

Usage:
    cd E:/TrendScope/backend
    python scripts/train_trend_classifier.py

This trains both:
  1. Logistic Regression (baseline)
  2. Random Forest (improved)

And saves evaluation metrics to:
  app/ml/models/trend_classifier_results.json
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo
from app.ml.trend_classifier import TrendClassifier
from app.ml.niche_filter import DB_GENRE_MAP


def train(genre: str = None, limit: int = 10000):
    """
    Train the TrendClassifier on historical video data.

    Parameters
    ----------
    genre : optional creator-profile genre (e.g. "Travel", "Gaming").
            Used for niche relevance scoring during training.
            Automatically mapped to actual DB genre labels.
    limit : max number of videos to use for training
    """
    app = create_app()
    with app.app_context():
        print(f"\n{'='*60}")
        print("  TrendScope — Trend Classifier Training")
        print(f"{'='*60}")
        print(f"  Genre filter : {genre or 'All genres'}")
        print(f"  Data limit   : {limit:,} videos")
        print(f"{'='*60}\n")

        # Load data
        print("[1/3] Loading data from database...")
        query = db.session.query(CleanVideo).filter(
            CleanVideo.view_count.isnot(None),
            CleanVideo.view_count > 0,
        )

        # Map creator genre → actual DB genre column values
        db_genres = DB_GENRE_MAP.get(genre, []) if genre else []
        if db_genres:
            print(f"      Mapping '{genre}' -> DB genres: {db_genres}")
            query = query.filter(CleanVideo.genre.in_(db_genres))
        elif genre:
            # Genre not in map — train on all data, still use genre for niche scoring
            print(f"      Note: '{genre}' not found as a DB genre. Training on all data.")
            print(f"      Niche relevance scoring will still use '{genre}' vocabulary.")

        videos = query.order_by(CleanVideo.trending_date.desc()).limit(limit).all()
        print(f"      Loaded {len(videos):,} videos.")

        if len(videos) < 50:
            print("ERROR: Not enough data. Need at least 50 videos.")
            return

        # Train
        print("\n[2/3] Training models...")
        classifier = TrendClassifier()
        results = classifier.train(videos, genre=genre)

        if "error" in results:
            print(f"ERROR: {results['error']}")
            return

        # Report
        print("\n[3/3] Evaluation Results")
        print(f"{'='*60}")
        print(f"  Model version  : {results.get('modelVersion', 'v1')}")
        print(f"  Split method   : {results.get('splitMethod', 'unknown')}")
        print(f"  Training set   : {results['trainingSize']:,} samples")
        print(f"  Test set       : {results['testSize']:,} samples")
        print(f"  Trend threshold: {results['trendThresholdViews']:,} views (top 25%)")
        print(f"  Positive rate  : {results['positiveRate']:.1%}")
        print(f"\n  Leakage note   : {results.get('dataLeakageNote', 'N/A')}")
        print(f"\n  Features used  : {', '.join(results.get('featuresUsed', []))}\n")

        b = results['baseline']
        r = results['improved']
        print(f"  {'Metric':<12} {'Logistic Reg':>14} {'Random Forest':>14}")
        print(f"  {'-'*42}")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']:
            print(f"  {metric:<12} {b[metric]:>14.4f} {r[metric]:>14.4f}")

        print(f"\n  Confusion Matrix (Random Forest):")
        cm = r.get('confusionMatrix', {})
        print(f"    TP (correct trending)    : {cm.get('TP', 'N/A')}")
        print(f"    TN (correct non-trending): {cm.get('TN', 'N/A')}")
        print(f"    FP (false trending alert): {cm.get('FP', 'N/A')}")
        print(f"    FN (missed trending)     : {cm.get('FN', 'N/A')}")

        imp = results['improvement']
        print(f"\n  Improvement (RF vs LR):")
        print(f"    F1 delta       : +{imp['f1_delta']:.4f}")
        print(f"    Accuracy delta : +{imp['accuracy_delta']:.4f}")
        print(f"    AUC delta      : +{imp['auc_delta']:.4f}")

        cv = results['crossValidation']
        print(f"\n  Cross-Validation (5-Fold, Random Forest):")
        print(f"    Mean F1  : {cv['meanF1']:.4f}")
        print(f"    Std F1   : {cv['stdF1']:.4f}  <- {cv['interpretation']}")

        print(f"\n  Feature Importance (Random Forest):")
        for feat, imp_val in results['featureImportance'].items():
            bar = '#' * int(imp_val * 40)
            print(f"    {feat:<28} {imp_val:.4f}  {bar}")

        print(f"\n{'='*60}")
        print("  Model saved to: app/ml/models/trend_classifier.pkl")
        print("  Results saved : app/ml/models/trend_classifier_results.json")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TrendScope Trend Classifier")
    parser.add_argument("--genre", type=str, default=None,
                        help="Genre to filter training data (e.g. Travel, Gaming)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max number of videos to use for training")
    args = parser.parse_args()
    train(genre=args.genre, limit=args.limit)
