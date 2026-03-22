
import requests
import json
from datetime import datetime, timedelta

def verify_api():
    print("--- Verifying Hybrid Model via API (localhost:5000) ---", flush=True)
    
    url = "http://localhost:5000/api/analytics/trend-strategy"
    
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    payload = {
        "region": "US",
        "startDate": start_date,
        "endDate": end_date,
        "useAdvanced": True,
        "modelType": "hybrid"
    }
    
    try:
        print(f" sending POST to {url} with modelType='hybrid'...", flush=True)
    
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60) # High timeout for first training
                break # Break out of the retry loop if successful
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    print(f"Connection refused, retrying ({i+1}/{max_retries})...", flush=True)
                    import time
                    time.sleep(2)
                else:
                    raise # Re-raise the exception if all retries failed
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ API SUCCESS!", flush=True)
            print("-" * 40)
            print(f"Strategy Engine: {data.get('detailedStrategy', {}).get('strategyEngine', 'unknown')}")
            # Check strategy text or keywords if visible
            print(f"Result Keys: {list(data.keys())}")
            print("-" * 40)
        else:
            print(f"\n❌ API FAILURE: {response.status_code}", flush=True)
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}", flush=True)
        print("Ensure the backend server is running on port 5000.")

if __name__ == "__main__":
    verify_api()
