"""Quick check: personalized strategy with auth token."""
import requests, json, time

BASE = "http://localhost:5000/api"

# Login (reuse test user)
print("Logging in...")
r = requests.post(f"{BASE}/auth/signup", json={
    "full_name": "Creator Tester",
    "email": "creator_quick_test@test.com",
    "password": "testpassword123",
}, timeout=10)
if r.status_code == 409:
    r = requests.post(f"{BASE}/auth/login", json={
        "email": "creator_quick_test@test.com",
        "password": "testpassword123",
    }, timeout=10)
token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"Token OK (user {r.json()['user']['id']})")

# Save profile
print("Saving profile...")
r = requests.post(f"{BASE}/onboarding/profile", json={
    "primaryGenre": "Gaming",
    "contentStyle": "Tutorials",
    "targetAudienceAge": "18-24",
    "targetRegion": "US",
    "creatorGoal": "increase_views",
}, headers=headers, timeout=10)
print(f"Profile: {r.status_code} — onboarding={r.json()['profile']['onboardingCompleted']}")

# Strategy call with retries
print("Calling personalized strategy (hybrid)...")
for attempt in range(3):
    try:
        r = requests.post(f"{BASE}/analytics/trend-strategy", json={
            "region": "US",
            "startDate": "2026-02-16",
            "endDate": "2026-03-18",
            "useAdvanced": True,
            "modelType": "hybrid",
        }, headers=headers, timeout=180)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Source: {data.get('strategySource')}")
            print(f"  Genre: {data.get('coreGenre')}")
            has_cp = "creatorProfile" in data.get("mlSignals", {})
            print(f"  Has creatorProfile: {has_cp}")
            strat = data.get("detailedStrategy", "")
            print(f"  Strategy: {strat[:300]}...")
            hooks = data.get("marketingHooks", [])
            print(f"  Hooks: {hooks}")
            with open("personalized_result.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            print("  Saved to personalized_result.json")
        else:
            print(f"  Error: {r.text[:200]}")
        break
    except requests.exceptions.ConnectionError as e:
        print(f"  Attempt {attempt+1} failed: {e}")
        time.sleep(3)
