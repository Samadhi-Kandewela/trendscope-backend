from datetime import datetime
from sqlalchemy.dialects.postgresql import ARRAY
from ..extensions import db
from ..utils.genre_mapping import CATEGORY_TO_UI_GENRE


class Video(db.Model):
    __tablename__ = "videos"

    id = db.Column(db.String, primary_key=True)           # video_id
    channel_id = db.Column(db.String, index=True)
    channel_title = db.Column(db.String)
    category_id = db.Column(db.Integer, index=True)       # video_category_id / category_id
    title = db.Column(db.String)
    description = db.Column(db.Text)
    default_thumbnail = db.Column(db.String)
    tags = db.Column(ARRAY(db.String))                    # video_tags
    view_count = db.Column(db.BigInteger)
    like_count = db.Column(db.BigInteger)
    dislike_count = db.Column(db.BigInteger)
    comment_count = db.Column(db.BigInteger)
    trending_country = db.Column(db.String(64), index=True)
    trending_date = db.Column(db.DateTime, index=True)
    published_at = db.Column(db.DateTime, index=True)
    source_dataset = db.Column(db.String(32))             # "2020" or "old" or "live_api"

    # Sentiment Analysis Fields
    sentiment_score = db.Column(db.Float)                 # avg sentiment of comments
    dominant_emotion = db.Column(db.String(32))           # prevailing emotion

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def genre(self) -> str:
        """Map YouTube category_id to a simplified UI genre."""
        return CATEGORY_TO_UI_GENRE.get(self.category_id, "Other")
