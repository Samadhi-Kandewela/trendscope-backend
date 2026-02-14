from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.video import Video
from ..models.clean_video import CleanVideo
from ..extensions import db

class DataMerger:
    """
    Handles the merging of Live API data into the existing video tables.
    Ensures 'source_dataset' is correctly tagged and prevents duplicates
    where appropriate.
    """

    def __init__(self, db_session: Session):
        self.session = db_session

    def merge_live_video(self, video_data: Dict[str, Any]) -> str:
        """
        Upserts a live video into the `videos` table.
        
        Rules:
        1. If video_id exists and source='2020' or 'old':
           - Update stats (view_count, etc.)
           - KEEP existing source_dataset (so we know it's also historical)
           - Update 'trending_date' to NOW if it's trending now.
        2. If video_id does not exist:
           - Insert new row with source_dataset='live_api'
        
        Returns: "inserted", "updated", or "skipped"
        """
        video_id = video_data.get("videoId")
        if not video_id:
            return "skipped"

        existing = self.session.query(Video).filter_by(id=video_id).first()
        
        now = datetime.utcnow()
        
        published_at = video_data.get("publishedAt")
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except ValueError:
                published_at = now

        if existing:
            # UPDATE existing
            existing.view_count = video_data.get("views", existing.view_count)
            existing.like_count = video_data.get("likes", existing.like_count)
            existing.comment_count = video_data.get("commentCount", existing.comment_count)
            existing.trending_date = now
            existing.trending_country = video_data.get("region_code", existing.trending_country)
            
            if not existing.title:
                existing.title = video_data.get("title")
            
            action = "updated"
        else:
            # INSERT new
            # Safely cast categoryId
            cat_id_raw = video_data.get("categoryId", 0)
            try:
                cat_id = int(cat_id_raw)
            except (ValueError, TypeError):
                cat_id = 0

            new_video = Video(
                id=video_id,
                channel_id=video_data.get("channelId"),
                channel_title=video_data.get("channel"),
                category_id=cat_id,
                title=video_data.get("title"),
                description=video_data.get("description"),
                default_thumbnail=video_data.get("thumbnail"),
                tags=video_data.get("tags", []),
                view_count=int(video_data.get("views", 0)),
                like_count=int(video_data.get("likes", 0)),
                comment_count=int(video_data.get("commentCount", 0)),
                trending_country=video_data.get("region_code"),
                trending_date=now,
                published_at=published_at,
                source_dataset="live_api"
            )
            self.session.add(new_video)
            action = "inserted"

        return action

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Returns composition of the dataset for the University Report.
        """
        # We need to commit/flush to see changes in a running transaction
        # self.session.flush()

        from sqlalchemy import func
        
        stats = self.session.query(
            Video.source_dataset, func.count(Video.id)
        ).group_by(Video.source_dataset).all()
        
        result = {
            "2020": 0,
            "old": 0,
            "live_api": 0,
            "total": 0
        }
        
        for source, count in stats:
            if source in result:
                result[source] = count
            result["total"] += count
            
            result["total"] += count
            
        return result

    def process_video_sentiment(self, video_id: str):
        """
        Fetches comments for a video, runs Gemini sentiment analysis,
        and saves the result to the Video and Comment tables.
        """
        from .youtube_service import get_video_comments
        from .sentiment_engine import analyze_comments
        from ..models.comment import Comment
        
        # 1. Fetch comments
        comments_data = get_video_comments(video_id, max_results=20)
        if not comments_data:
            return  # No comments or API error
            
        # 2. Extract text for analysis
        comment_texts = [c["text"] for c in comments_data]
        
        # 3. Analyze with Gemini
        score, emotion = analyze_comments(comment_texts)
        
        # 4. Update Video record
        video = self.session.query(Video).filter_by(id=video_id).first()
        if video:
            video.sentiment_score = score
            video.dominant_emotion = emotion
            
        # 5. Save Comments to DB
        for c in comments_data:
            # Check if comment exists
            existing_c = self.session.query(Comment).filter_by(id=c["id"]).first()
            if not existing_c:
                new_comment = Comment(
                    id=c["id"],
                    video_id=video_id,
                    author_display_name=c["author"],
                    text_display=c["text"],
                    like_count=c.get("likes", 0),
                    published_at=datetime.fromisoformat(c["publishedAt"].replace("Z", "+00:00")) if c.get("publishedAt") else datetime.utcnow()
                    # We could store individual sentiment per comment later if needed
                )
                self.session.add(new_comment)
        
        self.session.commit()
