from datetime import datetime
from flask import Blueprint, request, jsonify
from ..services.trend_service import get_genre_forecast, get_trend_strategy

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.get("/genres/near-term")
def near_term_genres():
    """
    Default Analytics section:
    GET /api/analytics/genres/near-term?region=Global&days=30
    """
    region = request.args.get("region", "Global")
    days = int(request.args.get("days", 30))

    result = get_genre_forecast(region=region, days=days)
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

    result = get_trend_strategy(region=region, start_date=start_date, end_date=end_date)
    if result is None:
        return jsonify({"error": "Not enough data to generate a trend strategy"}), 404

    return jsonify(result)
