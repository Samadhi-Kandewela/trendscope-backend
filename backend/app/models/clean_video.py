from datetime import datetime
from sqlalchemy.dialects.postgresql import ARRAY

from ..extensions import db


class CleanVideo(db.Model):
    """
    ML-ready cleaned representation of Video rows.

    Source of truth for:
      - genre forecast model
      - trend topic model
    """
    __tablename__ = "videos_clean"

    id = db.Column(db.Integer, primary_key=True)

    # Link back to original Video.id (YouTube video_id)
    video_id = db.Column(db.String, index=True)
    channel_id = db.Column(db.String, index=True)
    channel_title = db.Column(db.String)

    category_id = db.Column(db.Integer, index=True)   # final category_id after NLP fill
    genre = db.Column(db.String(32), index=True)      # resolved UI genre (Tech, Gaming, etc.)

    # Titles & tags (raw + cleaned)
    title = db.Column(db.String)
    title_clean = db.Column(db.Text)

    tags = db.Column(ARRAY(db.String))                # original tag array
    tags_text = db.Column(db.Text)                    # "tag1 tag2 tag3"
    tags_clean = db.Column(db.Text)                   # cleaned tags text

    # Description (cleaned)
    description_clean = db.Column(db.Text)

    # Country normalization
    trending_country_raw = db.Column(db.String(64))   # e.g., "United States", "New Zealand"
    trending_country_code = db.Column(db.String(2), index=True)  # e.g., "US", "NZ"

    trending_date = db.Column(db.DateTime, index=True)
    published_at = db.Column(db.DateTime, index=True)

    view_count = db.Column(db.BigInteger)
    like_count = db.Column(db.BigInteger)
    dislike_count = db.Column(db.BigInteger)
    comment_count = db.Column(db.BigInteger)

    source_dataset = db.Column(db.String(32))  # "2020" or "old"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
