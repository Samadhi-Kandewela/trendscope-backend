import sys
import os
import joblib

sys.path.append(os.getcwd())

def inspect_model_pickle():
    model_path = os.path.join("app", "ml", "models", "advanced_trend_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    try:
        data = joblib.load(model_path)
        print("Pickle loaded successfully.")
        print("Keys found:", data.keys())
        
        umap_obj = data.get("umap_model")
        hdbscan_obj = data.get("hdbscan_model")
        
        print("\n--- UMAP Object ---")
        print(f"Type: {type(umap_obj)}")
        # Check if it's an array or object
        if hasattr(umap_obj, 'transform'):
            print("Status: Can transform (GOOD)")
        else:
            print("Status: Cannot transform (likely an array) (BAD)")
            
        print("\n--- HDBSCAN Object ---")
        print(f"Type: {type(hdbscan_obj)}")
        # Check if prediction data is enabled
        if hasattr(hdbscan_obj, 'approximate_predict'):
            print("Status: Can predict (GOOD)")
        elif hasattr(hdbscan_obj, 'predict'):
            print("Status: Can predict (Standard) (GOOD)")
        else:
            print("Status: Cannot predict (BAD)")
            
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    inspect_model_pickle()
