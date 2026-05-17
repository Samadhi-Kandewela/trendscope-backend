"""
Comparison Script for TrendScope Models

This script loads BOTH:
1.  The Old "Baseline" Model (trend_topic_model.pkl)
2.  The New "Advanced" Model (advanced_trend_model.pkl)

It then runs predictions on a set of viral video titles to show the difference in clustering quality.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

# Import Old Model Logic (if available)
try:
    from app.ml.inference import load_trend_topic_model
except ImportError:
    print("Could not import old model loader. Make sure you are in the 'backend' directory.")
    sys.exit(1)

# Import New Model Libraries
try:
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
except ImportError:
    print("Please install dependencies: pip install sentence-transformers umap-learn hdbscan")
    sys.exit(1)

MODELS_DIR = os.path.join("app", "ml", "models")
OLD_MODEL_PATH = os.path.join(MODELS_DIR, "trend_topic_model.pkl")
NEW_MODEL_PATH = os.path.join(MODELS_DIR, "advanced_trend_model.pkl")

SAMPLE_TITLES = [
    "Minecraft 1.20 New Biome Update Review - Sniffer Mob",
    "Spicy Noodle Challenge MUST WATCH!! (Gone Wrong)",
    "Bitcoin Crash 2024? Crypto Market Analysis",
    "ASMR Sleep Routine (No Talking)",
    "How to make Sourdough Bread for Beginners",
    "GTA 6 Leaked Gameplay Reaction",
    "Try Not To Laugh Challenge - Best Vines 2024",
    "iPhone 16 Pro Max Unboxing & First Impressions",
    "Yoga for Back Pain Relief - 10 Minute Workout",
    "Gordon Ramsay Cooks Perfect Steak in 5 Minutes"
]

def load_new_model():
    if not os.path.exists(NEW_MODEL_PATH):
        print(f"ERROR: {NEW_MODEL_PATH} not found.")
        print("Please verify you copied the file from Colab.")
        return None
    
    print("Loading New Advanced Model...")
    try:
        data = joblib.load(NEW_MODEL_PATH)
        return data
    except Exception as e:
        print(f"Error loading new model: {e}")
        return None

def predict_old(model_data, texts):
    if not model_data:
        return ["N/A"] * len(texts)
    
    vectorizer = model_data["vectorizer"]
    kmeans = model_data["kmeans"]
    
    # Transform & Predict
    X = vectorizer.transform(texts)
    labels = kmeans.predict(X)
    return labels

def predict_new(model_data, texts):
    if not model_data:
        return ["N/A"] * len(texts)
    
    # 1. Embed
    embedder_name = model_data.get("embedding_model_name", "all-MiniLM-L6-v2")
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(texts, show_progress_bar=False)
    
    # 2. UMAP Transform
    umap_model = model_data["umap_model"]
    # UMAP transform might return varying dimensions based on n_components
    umap_embeddings = umap_model.transform(embeddings)
    
    # 3. HDBSCAN Predict
    hdbscan_model = model_data["hdbscan_model"]
    labels, strengths = hdbscan.approximate_predict(hdbscan_model, umap_embeddings)
    
    return labels

def run_comparison():
    print("-" * 60)
    print("TrendScope Model Comparison: Baseline vs Advanced")
    print("-" * 60)

    # 1. Load Old Model
    print("Loading Old Baseline Model...")
    old_data = load_trend_topic_model()
    if old_data:
        print("  -> Loaded (TF-IDF + K-Means)")
    else:
        print("  -> Failed to load Old Model")

    # 2. Load New Model
    new_data = load_new_model()
    if new_data:
        print("  -> Loaded (S-BERT + UMAP + HDBSCAN)")
    else:
        print("  -> Failed to load New Model (Skipping new predictions)")

    print("-" * 60)
    print(f"{'VIDEO TITLE':<50} | {'OLD (K-Means)':<15} | {'NEW (HDBSCAN)':<15}")
    print("-" * 60)

    # 3. Predict & Compare
    old_preds = predict_old(old_data, SAMPLE_TITLES)
    new_preds = []
    
    if new_data:
        try:
            new_preds = predict_new(new_data, SAMPLE_TITLES)
        except Exception as e:
            print(f"Error predicting with new model: {e}")
            new_preds = ["ERR"] * len(SAMPLE_TITLES)
    else:
        new_preds = ["N/A"] * len(SAMPLE_TITLES)

    # 4. Display
    for title, old_lbl, new_lbl in zip(SAMPLE_TITLES, old_preds, new_preds):
        t_short = (title[:47] + '..') if len(title) > 47 else title
        print(f"{t_short:<50} | Cluster {old_lbl:<7} | Cluster {new_lbl:<7}")

    print("-" * 60)
    print("Interpretation:")
    print("  - OLD: Probably groups all 'Gaming' together (Low Granularity)")
    print("  - NEW: Likely separates 'Minecraft' from 'GTA' (High Granularity)")
    print("  - -1 in NEW = Noise (Unique/Unclassifiable video)")
    print("-" * 60)

if __name__ == "__main__":
    run_comparison()
