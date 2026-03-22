# backend/app/models/creator_profile.py
from datetime import datetime
from ..extensions import db


class CreatorProfile(db.Model):
    __tablename__ = "creator_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False
    )

    # ── Onboarding Fields ──
    channel_url = db.Column(db.String(500), nullable=True)
    channel_id = db.Column(db.String(100), nullable=True)
    primary_genre = db.Column(db.String(50), nullable=True)
    content_style = db.Column(db.String(50), nullable=True)
    target_audience_age = db.Column(db.String(30), nullable=True)
    target_region = db.Column(db.String(10), nullable=True)
    creator_goal = db.Column(db.String(50), nullable=True)

    # ── Cached YouTube Channel Data ──
    channel_name = db.Column(db.String(255), nullable=True)
    subscriber_count = db.Column(db.Integer, nullable=True)
    total_views = db.Column(db.BigInteger, nullable=True)
    video_count = db.Column(db.Integer, nullable=True)
    channel_description = db.Column(db.Text, nullable=True)
    channel_thumbnail = db.Column(db.String(500), nullable=True)
    channel_data_fetched_at = db.Column(db.DateTime, nullable=True)

    # ── Status & Timestamps ──
    onboarding_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # ── Relationship ──
    user = db.relationship("User", backref=db.backref("creator_profile", uselist=False))

    # ── Serialization ──
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "channelUrl": self.channel_url,
            "channelId": self.channel_id,
            "primaryGenre": self.primary_genre,
            "contentStyle": self.content_style,
            "targetAudienceAge": self.target_audience_age,
            "targetRegion": self.target_region,
            "creatorGoal": self.creator_goal,
            "channelName": self.channel_name,
            "subscriberCount": self.subscriber_count,
            "totalViews": self.total_views,
            "videoCount": self.video_count,
            "channelDescription": self.channel_description,
            "channelThumbnail": self.channel_thumbnail,
            "channelDataFetchedAt": (
                self.channel_data_fetched_at.isoformat()
                if self.channel_data_fetched_at
                else None
            ),
            "onboardingCompleted": self.onboarding_completed,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
