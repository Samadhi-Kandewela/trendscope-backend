
from app.ml.lightweight_inference import HybridInferenceEngine
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_JSON = BASE_DIR / "app/ml/models/hybrid_model_params.json"

def test_engine():
    print(f"Loading engine from {MODEL_JSON}...")
    engine = HybridInferenceEngine(MODEL_JSON)
    
    if not engine.loaded:
        print("❌ Engine failed to load.")
        return

    print("✅ Engine loaded.")
    print(f"Vocab size: {len(engine.vocabulary)}")
    print(f"Centroids: {len(engine.centroids)}")
    
    texts = [
        "make money online fast", 
        "best makeup tutorial for beginners",
        "funny cat videos compilation",
        "how to cook pasta",
        "crypto bitcoin news"
    ]
    
    print("\nRunning prediction...")
    try:
        labels = engine.predict(texts)
        print(f"✅ Prediction successful: {labels}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
