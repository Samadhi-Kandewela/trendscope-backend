
import requests
import json
from datetime import datetime, timedelta

url = "http://localhost:5000/api/analytics/trend-strategy"
start_date = datetime.now().strftime("%Y-%m-%d")
end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

models = [
    ("LEGACY", {"region":"US","startDate":start_date,"endDate":end_date,"useAdvanced":False,"modelType":"legacy"}),
    ("HYBRID", {"region":"US","startDate":start_date,"endDate":end_date,"useAdvanced":True,"modelType":"hybrid"}),
]

output_lines = []
for name, payload in models:
    output_lines.append(f"\n=== {name} ===")
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            output_lines.append(f"STATUS:   {r.status_code}")
            output_lines.append(f"SOURCE:   {data.get('strategySource')}")
            output_lines.append(f"GENRE:    {data.get('coreGenre')}")
            output_lines.append(f"TREND:    {data.get('coreTrend')}")
            output_lines.append(f"KEYWORDS: {data.get('mlSignals', {}).get('topKeywords', [])}")
            output_lines.append(f"ENGINE:   {data.get('mlSignals', {}).get('strategyEngine', 'N/A')}")
            strat = data.get('detailedStrategy', '')
            output_lines.append(f"STRATEGY: {strat[:200] if strat else 'NONE'}...")
        else:
            output_lines.append(f"FAILED: HTTP {r.status_code}")
    except Exception as e:
        output_lines.append(f"ERROR: {e}")

result = "\n".join(output_lines)
print(result)

# Also write to file for inspection
with open("test_results.txt", "w") as f:
    f.write(result)
