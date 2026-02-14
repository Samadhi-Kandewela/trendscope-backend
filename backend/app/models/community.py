from datetime import datetime
from ..extensions import db

class Discussion(db.Model):
    __tablename__ = 'discussions'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=True)  # Markdown supported
    
    # "Instagram-style" Visual Fields
    media_url = db.Column(db.String(500), nullable=True)  # Image/chart URL
    linked_context = db.Column(db.JSON, nullable=True)    # {region: "US", date: "..."}
    
    author_username = db.Column(db.String(100), nullable=False, default="Anonymous")
    tags = db.Column(db.JSON, default=list)  # ["gaming", "strategy"]
    
    upvotes = db.Column(db.Integer, default=0)
    view_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    comments = db.relationship('DiscussionComment', backref='discussion', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "media_url": self.media_url,
            "linked_context": self.linked_context,
            "author": self.author_username,
            "tags": self.tags,
            "upvotes": self.upvotes,
            "views": self.view_count,
            "created_at": self.created_at.isoformat() + "Z",
            "comment_count": len(self.comments)
        }

class DiscussionComment(db.Model):
    __tablename__ = 'discussion_comments'

    id = db.Column(db.Integer, primary_key=True)
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussions.id'), nullable=False)
    
    body = db.Column(db.Text, nullable=False)
    author_username = db.Column(db.String(100), nullable=False, default="Anonymous")
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "discussion_id": self.discussion_id,
            "body": self.body,
            "author": self.author_username,
            "created_at": self.created_at.isoformat() + "Z"
        }

class DiscussionVote(db.Model):
    __tablename__ = 'discussion_votes'

    id = db.Column(db.Integer, primary_key=True)
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussions.id'), nullable=False)
    user_username = db.Column(db.String(100), nullable=False)
    vote_type = db.Column(db.String(10), default='up') # 'up' or 'down'

    # Ensure one vote per user per discussion
    __table_args__ = (db.UniqueConstraint('discussion_id', 'user_username', name='_user_discussion_uc'),)
