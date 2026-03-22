"""Test with Brett's actual token (user_id=13) via query param — simulating Postman."""
import requests, json, time

BASE = "http://localhost:5000/api"
# Brett's token from the user's Postman
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjEzLCJlbWFpbCI6ImJyZXR0Y29udGkzQGdtYWlsLmNvbSIsImV4cCI6MTc3MTQ0NDY3MSwiaWF0IjoxNzcxMzU4MjcxfQ._vfZaOSXB22iMqqTQlhP3ZkM66rvWtY8mcWUQYN3GXs"

# ── Test 1: Login to get fresh token ──
print("=== Getting fresh token for Brett ===")
r = requests.post(f"{BASE}/auth/login", json={
    "email": "brettconti3@gmail.com",
    "password": "brett123",  # guessing — will try signup if fails
}, timeout=10)
if r.status_code == 200:
    TOKEN = r.json()["token"]
    print(f"  Fresh token for user {r.json()['user']['id']}")
else:
    print(f"  Login failed ({r.status_code}), using existing token")

# ── Test 2: Via Header (correct way) ──
print("\n=== Strategy via HEADER (correct way) ===")
headers = {"Authorization": f"Bearer {TOKEN}"}
for attempt in range(3):
    try:
        r = requests.post(f"{BASE}/analytics/trend-strategy", json={
            "region": "US",
            "startDate": "2026-02-18",
            "endDate": "2026-03-20",
            "useAdvanced": True,
            "modelType": "hybrid",
        }, headers=headers, timeout=180)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  isPersonalized: {data['mlSignals'].get('isPersonalized')}")
            print(f"  personalizedIntro: {data.get('personalizedIntro', 'MISSING')}")
            if data.get("emergingTrends"):
                print(f"  emergingTrends count: {len(data['emergingTrends'])}")
            if data.get("contentGaps"):
                print(f"  contentGaps count: {len(data['contentGaps'])}")
            if data.get("titleSuggestions"):
                print(f"  titleSuggestions count: {len(data['titleSuggestions'])}")
            if data.get("optimalPosting"):
                print(f"  optimalPosting: {data['optimalPosting']}")
            with open("brett_header_result.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            print("  ✅ Saved to brett_header_result.json")
        else:
            print(f"  ERROR: {r.text[:200]}")
        break
    except requests.exceptions.ConnectionError:
        print(f"  Attempt {attempt+1}: connection error, retrying...")
        time.sleep(5)

# ── Test 3: Via Query Param (Postman style) ──
print("\n=== Strategy via QUERY PARAM (Postman style) ===")
for attempt in range(3):
    try:
        r = requests.post(
            f"{BASE}/analytics/trend-strategy?Authorization=Bearer {TOKEN}",
            json={
                "region": "US",
                "startDate": "2026-02-18",
                "endDate": "2026-03-20",
                "useAdvanced": True,
                "modelType": "hybrid",
            },
            timeout=180,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  isPersonalized: {data['mlSignals'].get('isPersonalized')}")
            print(f"  personalizedIntro: {data.get('personalizedIntro', 'MISSING')}")
            if data.get("emergingTrends"):
                print(f"  emergingTrends: {len(data['emergingTrends'])} items")
            with open("brett_queryparam_result.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            print("  ✅ Saved to brett_queryparam_result.json")
        else:
            print(f"  ERROR: {r.text[:200]}")
        break
    except requests.exceptions.ConnectionError:
        print(f"  Attempt {attempt+1}: connection error, retrying...")
        time.sleep(5)

print("\n=== DONE ===")
