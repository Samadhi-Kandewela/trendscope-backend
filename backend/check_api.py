
import requests
import json

url = "http://localhost:5000/api/analytics/trend-strategy"
payload = {
  "region": "US",
  "startDate": "2026-02-16",
  "endDate": "2026-03-16",
  "useAdvanced": False,  # Changed to False to prevent S-BERT loading crash
  "modelType": "hybrid"
}

try:
    print(f"Sending request to {url}...")
    r = requests.post(url, json=payload, timeout=60)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        # Check specific fields
        print(f"\nStrategy Source: {data.get('strategySource')}")
        print(f"Keywords: {data.get('mlSignals', {}).get('topKeywords')}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Exception: {e}")
