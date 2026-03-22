
import requests
import json
from datetime import datetime, timedelta

url = "http://localhost:5000/api/analytics/trend-strategy"
start_date = datetime.now().strftime("%Y-%m-%d")
end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

# Test Hybrid only (Legacy already verified)
payload = {
    "region": "US",
    "startDate": start_date,
    "endDate": end_date,
    "useAdvanced": True,
    "modelType": "hybrid"
}

try:
    r = requests.post(url, json=payload, timeout=120)
    data = r.json()
    with open("hybrid_result.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"STATUS: {r.status_code}")
    print(f"SOURCE: {data.get('strategySource')}")
    print(f"GENRE:  {data.get('coreGenre')}")
    print(f"TREND:  {data.get('coreTrend')}")
    print(f"ENGINE: {data.get('mlSignals',{}).get('strategyEngine','N/A')}")
    kw = data.get('mlSignals',{}).get('topKeywords',[])
    print(f"KEYWORDS: {kw}")
except Exception as e:
    print(f"ERROR: {e}")
