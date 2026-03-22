"""
End-to-end test: Creator Onboarding + Personalized Strategy
Steps:
  1. Signup a test user
  2. Save creator profile (Travel YouTuber)
  3. Verify profile GET
  4. Request personalized trend strategy (with auth)
  5. Request generic trend strategy (without auth) — regression check
"""
import requests
import json
from datetime import datetime, timedelta

BASE = "http://localhost:5000/api"

# ── 1. Signup ──
print("=== STEP 1: Signup ===")
signup_resp = requests.post(f"{BASE}/auth/signup", json={
    "full_name": "Test Creator",
    "email": f"test_onboard_{int(datetime.now().timestamp())}@test.com",
    "password": "testpassword123",
}, timeout=10)
print(f"  Status: {signup_resp.status_code}")
if signup_resp.status_code == 201:
    token = signup_resp.json()["token"]
    user_id = signup_resp.json()["user"]["id"]
    print(f"  User ID: {user_id}")
    print(f"  Token: {token[:30]}...")
elif signup_resp.status_code == 409:
    # Already exists, login instead
    print("  User exists, logging in...")
    login_resp = requests.post(f"{BASE}/auth/login", json={
        "email": signup_resp.json().get("email", "test@test.com"),
        "password": "testpassword123",
    }, timeout=10)
    token = login_resp.json()["token"]
    user_id = login_resp.json()["user"]["id"]
else:
    print(f"  ERROR: {signup_resp.text}")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

# ── 2. Get Enums ──
print("\n=== STEP 2: Get Onboarding Enums ===")
enums_resp = requests.get(f"{BASE}/onboarding/enums", timeout=10)
print(f"  Status: {enums_resp.status_code}")
if enums_resp.status_code == 200:
    enums = enums_resp.json()
    print(f"  Genres: {enums['genres'][:5]}...")
    print(f"  Styles: {enums['contentStyles'][:5]}...")

# ── 3. Save Profile ──
print("\n=== STEP 3: Save Creator Profile ===")
profile_resp = requests.post(f"{BASE}/onboarding/profile", json={
    "channelUrl": "https://www.youtube.com/@TravelWithMe",
    "primaryGenre": "Travel",
    "contentStyle": "Vlogs",
    "targetAudienceAge": "18-24",
    "targetRegion": "US",
    "creatorGoal": "grow_subscribers",
}, headers=headers, timeout=10)
print(f"  Status: {profile_resp.status_code}")
if profile_resp.status_code == 201:
    profile = profile_resp.json()["profile"]
    print(f"  Genre: {profile['primaryGenre']}")
    print(f"  Style: {profile['contentStyle']}")
    print(f"  Goal: {profile['creatorGoal']}")
    print(f"  Onboarding Complete: {profile['onboardingCompleted']}")
else:
    print(f"  ERROR: {profile_resp.text}")

# ── 4. Get Profile ──
print("\n=== STEP 4: Get Profile ===")
get_resp = requests.get(f"{BASE}/onboarding/profile", headers=headers, timeout=10)
print(f"  Status: {get_resp.status_code}")
if get_resp.status_code == 200:
    print(f"  Onboarding Completed: {get_resp.json().get('onboardingCompleted')}")

# ── 5. Personalized Strategy (WITH auth) ──
print("\n=== STEP 5: Personalized Trend Strategy (Hybrid + Auth) ===")
start_date = datetime.now().strftime("%Y-%m-%d")
end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
strat_resp = requests.post(f"{BASE}/analytics/trend-strategy", json={
    "region": "US",
    "startDate": start_date,
    "endDate": end_date,
    "useAdvanced": True,
    "modelType": "hybrid",
}, headers=headers, timeout=120)
print(f"  Status: {strat_resp.status_code}")
if strat_resp.status_code == 200:
    data = strat_resp.json()
    print(f"  Source: {data.get('strategySource')}")
    print(f"  Genre:  {data.get('coreGenre')}")
    print(f"  Trend:  {data.get('coreTrend')}")
    kw = data.get('mlSignals', {}).get('topKeywords', [])
    print(f"  Keywords: {kw}")
    has_profile = "creatorProfile" in data.get("mlSignals", {})
    print(f"  Has Creator Profile in Signals: {has_profile}")
    if has_profile:
        cp = data["mlSignals"]["creatorProfile"]
        print(f"  Creator Genre: {cp.get('primaryGenre')}")
    strat = data.get("detailedStrategy", "")
    print(f"  Strategy (first 200 chars): {strat[:200]}...")
    with open("personalized_result.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
else:
    print(f"  ERROR: {strat_resp.text[:300]}")

# ── 6. Generic Strategy (WITHOUT auth) — regression check ──
print("\n=== STEP 6: Generic Strategy (Legacy, No Auth) ===")
generic_resp = requests.post(f"{BASE}/analytics/trend-strategy", json={
    "region": "US",
    "startDate": start_date,
    "endDate": end_date,
    "useAdvanced": False,
    "modelType": "legacy",
}, timeout=120)
print(f"  Status: {generic_resp.status_code}")
if generic_resp.status_code == 200:
    data = generic_resp.json()
    print(f"  Source: {data.get('strategySource')}")
    print(f"  Genre:  {data.get('coreGenre')}")
    has_profile = "creatorProfile" in data.get("mlSignals", {})
    print(f"  Has Creator Profile: {has_profile} (should be False)")

print("\n=== ALL TESTS COMPLETE ===")
