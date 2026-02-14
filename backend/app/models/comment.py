from datetime import datetime
from ..extensions import db

class Comment(db.Model):
    __tablename__ = "comments"

    id = db.Column(db.String, primary_key=True)  # YouTube comment ID
    video_id = db.Column(db.String, db.ForeignKey('videos.id'), nullable=False, index=True)
    author_display_name = db.Column(db.String)
    text_display = db.Column(db.Text)
    like_count = db.Column(db.Integer, default=0)
    published_at = db.Column(db.DateTime, index=True)
    
    # Sentiment fields (calculated later)
    sentiment_score = db.Column(db.Float)  # -1.0 to 1.0 from Gemini
    emotion_label = db.Column(db.String(32)) # e.g. "Joy", "Anger"

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "videoId": self.video_id,
            "author": self.author_display_name,
            "text": self.text_display,
            "likes": self.like_count,
            "sentiment": self.sentiment_score,
            "emotion": self.emotion_label
        }
