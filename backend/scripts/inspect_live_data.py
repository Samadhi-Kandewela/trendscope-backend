from app import create_app
from app.extensions import db
from app.models.video import Video

app = create_app()

with app.app_context():
    print("--- Inspecting 'live_api' Dataset ---")
    # Fetch first 5 videos from the live_api source
    live_videos = db.session.query(Video).filter_by(source_dataset='live_api').limit(5).all()
    
    if not live_videos:
        print("No videos found in 'live_api' dataset.")
    else:
        print(f"Found {len(live_videos)} samples (showing top 5):")
        for v in live_videos:
            print(f"\n[ID: {v.id}]")
            print(f"Title: {v.title}")
            print(f"Channel: {v.channel_title}")
            print(f"Views: {v.view_count}")
            print(f"Region: {v.trending_country}")
            print(f"Trending Date: {v.trending_date}")

    print("\n-----------------------------------")
    print("Location: 'videos' table in PostgreSQL database.")
