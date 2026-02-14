from app import create_app
from app.services.youtube_service import get_trending_videos

app = create_app()

with app.app_context():
    try:
        print("Fetching Gaming videos for US...")
        videos = get_trending_videos("US", "Gaming", limit=5)
        print(f"Success! Fetched {len(videos)} videos.")
        for v in videos:
            print(f"- {v['title']} ({v['views']} views)")
    except Exception as e:
        print(f"API Call Failed: {e}")
