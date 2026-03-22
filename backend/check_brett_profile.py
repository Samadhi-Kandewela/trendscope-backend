"""Check if user 13 (Brett) has a creator profile and create one if not."""
import sys
sys.path.insert(0, ".")
from app import create_app
from app.extensions import db
from app.models.creator_profile import CreatorProfile
from app.models.user import User

app = create_app()
with app.app_context():
    user = User.query.get(13)
    if user:
        print(f"User 13: {user.full_name} ({user.email})")
        profile = CreatorProfile.query.filter_by(user_id=13).first()
        if profile:
            print(f"  Profile exists: genre={profile.primary_genre}, style={profile.content_style}")
            print(f"  Onboarding completed: {profile.onboarding_completed}")
            print(f"  Goal: {profile.creator_goal}")
        else:
            print("  ❌ NO creator profile found! Creating one...")
            profile = CreatorProfile(
                user_id=13,
                channel_url="https://www.youtube.com/@BrettConti",
                primary_genre="Travel",
                content_style="Vlogs",
                target_audience_age="25-34",
                target_region="US",
                creator_goal="increase_views",
                onboarding_completed=True,
            )
            db.session.add(profile)
            db.session.commit()
            print("  ✅ Profile created! onboarding_completed=True")
    else:
        print("User 13 not found!")
    
    # Show all profiles
    print("\n--- All Creator Profiles ---")
    for p in CreatorProfile.query.all():
        u = User.query.get(p.user_id)
        name = u.full_name if u else "?"
        print(f"  user_id={p.user_id} ({name}): genre={p.primary_genre}, completed={p.onboarding_completed}")
