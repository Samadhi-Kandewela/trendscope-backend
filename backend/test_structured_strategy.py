"""Test the new structured personalized strategy + title optimizer."""
import requests, json, time

BASE = "http://localhost:5000/api"

# ── Login and setup ──
print("=== Login ===")
r = requests.post(f"{BASE}/auth/signup", json={
    "full_name": "Test V2",
    "email": "test_v2_structured@test.com",
    "password": "testpassword123",
}, timeout=10)
if r.status_code == 409:
    r = requests.post(f"{BASE}/auth/login", json={
        "email": "test_v2_structured@test.com",
        "password": "testpassword123",
    }, timeout=10)
token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"  User {r.json()['user']['id']} OK")

# ── Save profile (Travel / Vlogs) ──
print("\n=== Save Profile ===")
r = requests.post(f"{BASE}/onboarding/profile", json={
    "primaryGenre": "Travel",
    "contentStyle": "Vlogs",
    "targetAudienceAge": "25-34",
    "targetRegion": "US",
    "creatorGoal": "increase_views",
}, headers=headers, timeout=10)
print(f"  Status: {r.status_code} — onboarding={r.json()['profile']['onboardingCompleted']}")

# ── Personalized Strategy ──
print("\n=== Personalized Strategy (Travel Vlogs) ===")
for attempt in range(3):
    try:
        r = requests.post(f"{BASE}/analytics/trend-strategy", json={
            "region": "US",
            "startDate": "2026-02-17",
            "endDate": "2026-03-19",
            "useAdvanced": True,
            "modelType": "hybrid",
        }, headers=headers, timeout=180)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"\n  📌 Intro: {data.get('personalizedIntro', 'N/A')}")
            
            print("\n  📈 Emerging Trends:")
            for t in data.get("emergingTrends", []):
                print(f"    • {t.get('trend')} ({t.get('signal')})")
                print(f"      → {t.get('relevanceToYou')}")
            
            print("\n  🎯 Recommended Angles:")
            for a in data.get("recommendedAngles", []):
                print(f"    • {a.get('angle')}")
                print(f"      Why: {a.get('why')}")
            
            print("\n  🔍 Content Gaps:")
            for g in data.get("contentGaps", []):
                print(f"    • {g.get('topic')}")
                print(f"      {g.get('insight')}")
            
            print("\n  ✏️  Title Suggestions:")
            for ts in data.get("titleSuggestions", []):
                print(f"    • Draft: \"{ts.get('draft')}\"")
                print(f"      Optimized: \"{ts.get('optimized')}\"")
                print(f"      Why: {ts.get('whyBetter')}")
            
            posting = data.get("optimalPosting", {})
            if posting:
                print(f"\n  ⏰ Optimal Posting: {posting.get('bestDays')} at {posting.get('bestTime')}")
                print(f"     Reason: {posting.get('reason')}")
            
            print(f"\n  🏷  Marketing Hooks: {data.get('marketingInsights', [])}")
            print(f"\n  📝 Summary: {data.get('detailedStrategy', '')[:200]}...")
            
            with open("structured_result.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            print("\n  ✅ Saved to structured_result.json")
        else:
            print(f"  ERROR: {r.text[:300]}")
        break
    except requests.exceptions.ConnectionError as e:
        print(f"  Attempt {attempt+1} failed: connection error")
        time.sleep(5)

# ── Title Optimizer ──
print("\n\n=== Title Optimizer ===")
for attempt in range(3):
    try:
        r = requests.post(f"{BASE}/analytics/title-optimizer", json={
            "draftTitle": "My Trip to Japan",
            "region": "US",
        }, headers=headers, timeout=60)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"\n  Original: \"{data.get('originalTitle')}\"")
            for s in data.get("suggestions", []):
                print(f"    → \"{s.get('optimizedTitle')}\"")
                print(f"      Why: {s.get('whyBetter')}")
                print(f"      Keywords: {s.get('trendingKeywordsUsed')}")
            print(f"\n  Tips: {data.get('tips', [])}")
            with open("title_optimizer_result.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            print("  ✅ Saved to title_optimizer_result.json")
        else:
            print(f"  ERROR: {r.text[:300]}")
        break
    except requests.exceptions.ConnectionError as e:
        print(f"  Attempt {attempt+1} failed: connection error")
        time.sleep(5)

print("\n=== ALL TESTS COMPLETE ===")
