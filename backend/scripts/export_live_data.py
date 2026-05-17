import sys
import os
import csv
from datetime import datetime

# Add backend to path to import app
sys.path.append(os.getcwd())

from app import create_app, db
from app.models.video import Video

def export_live_data():
    app = create_app()
    with app.app_context():
        print("--- Exporting Live Data to CSV ---")
        
        # Query only live data
        videos = db.session.query(Video).filter(Video.source_dataset == 'live_api').all()
        
        if not videos:
            print("No live data found to export.")
            return

        filename = "live_data_export.csv"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write Header
            writer.writerow(['video_id', 'title', 'tags', 'view_count', 'like_count', 'comment_count', 'trending_date'])
            
            # Write Rows
            count = 0
            for v in videos:
                # Format tags as space-separated string for compatibility with notebook
                tags_str = " ".join(v.tags) if v.tags else ""
                trending = v.trending_date.isoformat() if v.trending_date else ""
                
                writer.writerow([
                    v.id, 
                    v.title, 
                    tags_str, 
                    v.view_count, 
                    v.like_count, 
                    v.comment_count, 
                    trending
                ])
                count += 1
                
        print(f"SUCCESS: Exported {count} videos to {filepath}")

if __name__ == "__main__":
    export_live_data()
