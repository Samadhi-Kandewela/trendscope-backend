
import requests
import json
from datetime import datetime, timedelta

def verify_legacy():
    print("--- Verifying Legacy Model (Standard K-Means) ---", flush=True)
    
    url = "http://localhost:5000/api/analytics/trend-strategy"
    
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    payload = {
        "region": "US",
        "startDate": start_date,
        "endDate": end_date,
        "useAdvanced": False,  # Logic: This triggers the baseline Path A
        "modelType": "legacy"  # Logic: Legacy mode
    }
    
    print("Request Body:")
    print(json.dumps(payload, indent=2))
    
    try:
        print(f"\nSending POST to {url}...", flush=True)
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ API SUCCESS!", flush=True)
            print("-" * 40)
            print(f"Strategy Engine: {data.get('detailedStrategy', {}).get('strategyEngine', 'unknown')}")
            # The legacy path usually falls back to heuristic if no strict ML model is loaded or trained, 
            # or uses the simple TF-IDF+Kmeans if available.
            
            print(f"Top Keywords: {data.get('mlSignals', {}).get('topKeywords')}")
            print("-" * 40)
        else:
            print(f"\n❌ API FAILURE: {response.status_code}", flush=True)
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}", flush=True)
        print("Ensure the backend server is running on port 5000.")

if __name__ == "__main__":
    verify_legacy()
