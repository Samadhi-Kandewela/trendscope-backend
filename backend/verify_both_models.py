
import requests
import json
from datetime import datetime, timedelta

url = "http://localhost:5000/api/analytics/trend-strategy"
start_date = datetime.now().strftime("%Y-%m-%d")
end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

# ---- TEST 1: LEGACY MODEL ----
print("TEST 1: LEGACY MODEL")

legacy_payload = {
    "region": "US",
    "startDate": start_date,
    "endDate": end_date,
    "useAdvanced": False,
    "modelType": "legacy"
}

try:
    r = requests.post(url, json=legacy_payload, timeout=60)
    if r.status_code == 200:
        data = r.json()
        print(f"  STATUS: OK ({r.status_code})")
        print(f"  SOURCE: {data.get('strategySource')}")
        print(f"  GENRE:  {data.get('coreGenre')}")
        print(f"  TREND:  {data.get('coreTrend')}")
        kw = data.get('mlSignals', {}).get('topKeywords', [])
        print(f"  KEYWORDS: {kw}")
    else:
        print(f"  FAILED: HTTP {r.status_code}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---- TEST 2: HYBRID MODEL ----
print("\nTEST 2: HYBRID MODEL")

hybrid_payload = {
    "region": "US",
    "startDate": start_date,
    "endDate": end_date,
    "useAdvanced": True,
    "modelType": "hybrid"
}

try:
    r = requests.post(url, json=hybrid_payload, timeout=60)
    if r.status_code == 200:
        data = r.json()
        print(f"  STATUS: OK ({r.status_code})")
        print(f"  SOURCE: {data.get('strategySource')}")
        print(f"  GENRE:  {data.get('coreGenre')}")
        print(f"  TREND:  {data.get('coreTrend')}")
        kw = data.get('mlSignals', {}).get('topKeywords', [])
        print(f"  KEYWORDS: {kw}")
    else:
        print(f"  FAILED: HTTP {r.status_code}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDONE")
