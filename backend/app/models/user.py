# backend/app/models/user.py
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

from ..extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    # Basic profile
    full_name = db.Column(db.String(120), nullable=False)

    # Auth fields
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # -------- Password helpers --------
    def set_password(self, raw_password: str) -> None:
        """Hash and store the user's password."""
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Return True if provided password matches the stored hash."""
        return check_password_hash(self.password_hash, raw_password)

    # -------- Serialization helper --------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
