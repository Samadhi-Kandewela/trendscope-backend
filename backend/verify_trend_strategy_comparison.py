
import json
import requests
from datetime import datetime
from app import create_app

def run_comparison():
    app = create_app()
    client = app.test_client()

    print("-" * 60)
    print("Trend Strategy Model Comparison (API Test)")
    print("-" * 60)

    # Payload
    payload_base = {
        "region": "US",
        "startDate": "2024-01-01",
        "endDate": "2024-02-01"
    }

    # 1. Old Model (Default)
    print("\n[1] Testing OLD Model (K-Means)...")
    try:
        resp_old = client.post("/api/analytics/trend-strategy", json=payload_base)
        if resp_old.status_code == 200:
            data_old = resp_old.get_json()
            strategy_old = data_old.get("detailedStrategy", "No strategy found")
            hooks_old = data_old.get("marketingHooks", [])
            print(f"  -> Success!")
            print(f"  -> Strategy Preview: {strategy_old[:100]}...")
            print(f"  -> Hooks: {hooks_old}")
        else:
            print(f"  -> Failed: {resp_old.status_code} - {resp_old.text}")
    except Exception as e:
        print(f"  -> Error: {e}")

    # 2. New Model (Advanced)
    print("\n[2] Testing NEW Model (S-BERT + UMAP)...")
    payload_new = payload_base.copy()
    payload_new["useAdvanced"] = True
    
    try:
        resp_new = client.post("/api/analytics/trend-strategy", json=payload_new)
        if resp_new.status_code == 200:
            data_new = resp_new.get_json()
            strategy_new = data_new.get("detailedStrategy", "No strategy found")
            hooks_new = data_new.get("marketingHooks", [])
            print(f"  -> Success!")
            print(f"  -> Strategy Preview: {strategy_new[:100]}...")
            print(f"  -> Hooks: {hooks_new}")
        else:
            print(f"  -> Failed: {resp_new.status_code} - {resp_new.text}")
    except Exception as e:
        print(f"  -> Error: {e}")

    print("-" * 60)

if __name__ == "__main__":
    run_comparison()
