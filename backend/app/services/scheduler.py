import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import current_app
from .youtube_service import fetch_all_trending_videos
from .data_manager import DataMerger
from ..extensions import db

scheduler = BackgroundScheduler()

def ingest_daily_trends():
    """
    Background job to fetch and merge live trending videos.
    """
    # We need to manually push an app context because this runs in a separate thread
    # However, we don't have easy access to 'app' object here unless we use a factory pattern strictly.
    # A common pattern with Flask-APScheduler or simple APScheduler is to pass the app app.
    # But here, let's assume we are called from within the running app or we handle context carefully.
    
    # Actually, the best way for a simple Flask app is to rely on current_app proxy being available 
    # IF the scheduler is started with the app.
    # But standard BackgroundScheduler runs in a thread. We need the app object.
    # We'll solve this by importing create_app? No, circular import.
    
    # Solution: We will attach the 'app' instance to the job or scheduler when initializing.
    pass

def run_ingestion(app):
    """
    The actual logic, accepting 'app' to create context.
    """
    with app.app_context():
        current_app.logger.info("Starting Daily Ingestion Job...")
        merger = DataMerger(db.session)
        
        # 1. Define target regions (can be config driven)
        regions = ["US", "GB", "IN", "CA", "AU"] 
        
        total_inserted = 0
        total_updated = 0
        
        for region in regions:
            current_app.logger.info(f"Fetching for region: {region}")
            videos = fetch_all_trending_videos(region)
            
            for v_data in videos:
                action = merger.merge_live_video(v_data)
                if action == "inserted":
                    total_inserted += 1
                elif action == "updated":
                    total_updated += 1
            
            # Commit after each region to save progress
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                current_app.logger.error(f"Commit failed for region {region}: {e}")
                
        current_app.logger.info(f"Ingestion Complete. Inserted: {total_inserted}, Updated: {total_updated}")

def start_scheduler(app):
    """
    Initialize and start the scheduler.
    """
    if not scheduler.running:
        # Add job to run every 24 hours
        # We pass 'app' as an argument to the job function
        scheduler.add_job(
            func=run_ingestion,
            trigger="interval",
            hours=24,
            args=[app],
            id="daily_ingestion_job",
            replace_existing=True
        )
        scheduler.start()
        app.logger.info("Scheduler started with daily ingestion job.")
        
        # Shut down scheduler when exiting the app
        atexit.register(lambda: scheduler.shutdown())
