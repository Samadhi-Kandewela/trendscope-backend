import sys
import os
from sqlalchemy import func

# Add backend to path to import app
sys.path.append(os.getcwd())

from app import create_app, db
from app.models.video import Video

def check_live_growth():
    app = create_app()
    with app.app_context():
        print("--- Checking Live Data Growth ---")
        
        # 1. Total Count
        total_live = db.session.query(Video).filter(Video.source_dataset == 'live_api').count()
        print(f"Total 'live_api' videos in DB: {total_live}")
        
        # 2. Daily Breakdown (if we use trending_date or created_at)
        # Assuming 'trending_date' is populated for live videos
        daily_counts = db.session.query(
            func.date(Video.trending_date), 
            func.count(Video.id)
        ).filter(
            Video.source_dataset == 'live_api'
        ).group_by(
            func.date(Video.trending_date)
        ).order_by(
            func.date(Video.trending_date)
        ).all()
        
        if daily_counts:
            print("\nDaily Ingestion Breakdown:")
            for date, count in daily_counts:
                print(f"  {date}: {count} videos")
        else:
            print("\nNo daily breakdown available (dates might be null or all same day).")

if __name__ == "__main__":
    check_live_growth()
