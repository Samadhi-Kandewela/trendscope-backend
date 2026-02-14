import logging
import sys
from app import create_app
from app.extensions import db
from app.services.data_manager import DataMerger
from app.models.video import Video
from app.models.comment import Comment

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = create_app()

with app.app_context():
    print("--- Verifying Sentiment Analysis ---")
    
    # 1. Find a target video (prefer live_api)
    video = db.session.query(Video).filter_by(source_dataset='live_api').first()
    if not video:
        print("No live_api video found. Trying any video...")
        video = db.session.query(Video).first()
        
    if not video:
        print("No videos found in DB.")
        sys.exit(1)
        
    print(f"Target Video: {video.title} (ID: {video.id})")
    
    # 2. Run functionality
    merger = DataMerger(db.session)
    print("Running process_video_sentiment...")
    try:
        merger.process_video_sentiment(video.id)
    except Exception as e:
        import traceback
        with open("sentiment_error.txt", "w") as f:
            f.write(f"Error processing sentiment: {e}\n")
            traceback.print_exc(file=f)
        sys.exit(1)
    
    # Verify results
    db.session.refresh(video)
    
    with open("sentiment_result.txt", "w") as f:
        f.write(f"Analyzed Video: {video.title} ({video.id})\n")
        f.write(f"Sentiment Score: {video.sentiment_score}\n")
        f.write(f"Dominant Emotion: {video.dominant_emotion}\n")
        
        comments = db.session.query(Comment).filter_by(video_id=video.id).all()
        f.write(f"Saved Comments: {len(comments)}\n")
        if comments:
            f.write(f"Sample Comment: {comments[0].text_display[:50]}...\n")
            
    print("Verification complete. Check sentiment_result.txt")
