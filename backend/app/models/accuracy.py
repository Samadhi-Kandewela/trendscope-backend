from datetime import datetime
from ..extensions import db

class AccuracyLog(db.Model):
    __tablename__ = 'accuracy_logs'

    id = db.Column(db.Integer, primary_key=True)
    log_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # "historical_backtest" or "live_validation"
    log_type = db.Column(db.String(50), nullable=False)
    
    # The calculated score (0.0 to 1.0)
    accuracy_score = db.Column(db.Float, nullable=False)
    
    # JSON details (e.g. {"precision": 0.82, "recall": 0.75, "matched_keywords": [...]})
    details = db.Column(db.JSON, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "log_date": self.log_date.isoformat(),
            "log_type": self.log_type,
            "accuracy_score": self.accuracy_score,
            "details": self.details or {}
        }
