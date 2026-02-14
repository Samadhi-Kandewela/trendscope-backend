from datetime import datetime
from flask import Blueprint, request, jsonify
from ..services.trend_service import get_genre_forecast, get_trend_strategy

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.get("/genres/near-term")
def near_term_genres():
    """
    Default Analytics section:
    GET /api/analytics/genres/near-term?region=Global&days=30
    OR
    GET /api/analytics/genres/near-term?region=US&startDate=2026-04-01&endDate=2026-04-30
    """
    region = request.args.get("region", "Global")
    days = int(request.args.get("days", 30))
    
    start_str = request.args.get("startDate")
    end_str = request.args.get("endDate")
    
    start_date = None
    end_date = None
    
    if start_str and end_str:
        try:
            start_date = datetime.fromisoformat(start_str)
            end_date = datetime.fromisoformat(end_str)
        except ValueError:
            pass # Fallback to default logic if dates invalid

    result = get_genre_forecast(region=region, days=days, start_date=start_date, end_date=end_date)
    return jsonify(result)


@analytics_bp.post("/trend-strategy")
def trend_strategy():
    """
    Trend Strategy for [Region] and [startDate, endDate].
    POST /api/analytics/trend-strategy
    Body:
    {
      "region": "US",
      "startDate": "2025-12-05",
      "endDate": "2026-01-05"
    }
    """
    data = request.get_json() or {}
    region = data.get("region", "Global")
    start_str = data.get("startDate")
    end_str = data.get("endDate")

    if not start_str or not end_str:
        return jsonify({"error": "startDate and endDate are required"}), 400

    try:
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
    except ValueError:
        return jsonify({"error": "Dates must be in ISO format: YYYY-MM-DD"}), 400

    use_advanced = data.get("useAdvanced", False)

    result = get_trend_strategy(
        region=region,
        start_date=start_date,
        end_date=end_date,
        use_advanced_model=use_advanced
    )
    if result is None:
        return jsonify({"error": "Not enough data to generate a trend strategy"}), 404

    return jsonify(result)


@analytics_bp.route('/viral-potential', methods=['GET'])
def get_viral_potential():
    """
    Predicts the viral potential of a video based on current metrics.
    Query Params: views, likes, comments, hours_since_upload
    Returns: JSON with prediction and probability label.
    """
    try:
        current_views = int(request.args.get('views', 0))
        likes = int(request.args.get('likes', 0))
        comments = int(request.args.get('comments', 0))
        hours = int(request.args.get('hours_since_upload', 24))
        
        # Simple rule: if hours < 0, default to 1 to avoid div/0 or weird logic
        if hours < 1: hours = 1
        
        # Convert hours to days roughly for the model which uses 'days_since_upload'
        days = hours / 24.0
        
        from ..ml.viral_velocity import ViralVelocityModel
        model = ViralVelocityModel()
        model.load() # Will train if missing
        
        result = model.predict(current_views, likes, comments, days)
        
        if not result:
            return jsonify({"error": "Model failed to predict"}), 500
            
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Viral prediction error: {e}")
        return jsonify({"error": str(e)}), 500
