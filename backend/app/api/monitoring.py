from flask import Blueprint, jsonify
from ..extensions import db
from ..models.accuracy import AccuracyLog
from ..ml.validation import ModelValidator

monitoring_bp = Blueprint('monitoring', __name__)

@monitoring_bp.route('/accuracy', methods=['GET'])
def get_accuracy_metrics():
    """
    Returns the latest accuracy scores for Historical and Live validation.
    Also triggers a fresh Live Validation check on demand (for demo purposes).
    """
    validator = ModelValidator()
    
    # 1. Trigger Live Check (Real-time)
    live_result = validator.run_live_accuracy_check()
    
    # 1b. Trigger Clustering Quality Check (Real-time sample)
    clustering_result = validator.run_clustering_quality_check()
    
    # 2. Fetch latest logs
    historical_log = db.session.query(AccuracyLog).filter_by(
        log_type='historical_backtest'
    ).order_by(AccuracyLog.log_date.desc()).first()
    
    live_log = db.session.query(AccuracyLog).filter_by(
        log_type='live_validation'
    ).order_by(AccuracyLog.log_date.desc()).first()
    
    clustering_log = db.session.query(AccuracyLog).filter_by(
        log_type='clustering_quality'
    ).order_by(AccuracyLog.log_date.desc()).first()
    
    return jsonify({
        "metrics": [
            {
                "label": "Historical Baseline Accuracy",
                "value": f"{historical_log.accuracy_score*100:.1f}%" if historical_log else "N/A",
                "description": "How accurately the model predicts past trends (2020 dataset) using a held-out test set.",
                "status": "Healthy" if historical_log and historical_log.accuracy_score > 0.8 else "Needs Attention"
            },
            {
                "label": "Real-Time Live Accuracy",
                "value": f"{live_log.accuracy_score*100:.1f}%" if live_log else "N/A",
                "description": "Percentage of predicted trend keywords that appear in TODAY'S actual trending videos.",
                "status": "Excellent" if live_log and live_log.accuracy_score > 0.9 else "Good"
            },
            {
                "label": "Clustering Separation (Silhouette)",
                "value": f"{clustering_log.accuracy_score:.2f}" if clustering_log else "N/A",
                "description": "Measures how distinct the trend clusters are (-1 to +1). Higher is better.",
                "status": "Excellent" if clustering_log and clustering_log.accuracy_score > 0.5 else "Good" if clustering_log and clustering_log.accuracy_score > 0.2 else "Weak"
            }
        ],
        "latest_live_details": live_result,
        "latest_clustering_details": clustering_result
    })
