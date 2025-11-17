from flask import Blueprint, request, jsonify
from ..services.youtube_service import get_keyword_analysis
from ..services.trend_service import normalize_region # Re-using normalization utility
from typing import Dict, Any, Optional

# Create the Blueprint instance for Explorer routes
explorer_bp = Blueprint("explorer", __name__)

@explorer_bp.post("/explore-keywords")
def explore_keywords():
    """
    POST /api/explorer/explore-keywords
    Handles keyword search, competitive analysis, and tag suggestions for the UI.
    """
    data: Dict[str, Any] = request.get_json() or {}
    keyword: Optional[str] = data.get("keyword")
    region_raw: str = data.get("region", "Global")

    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400

    # Normalize the region code (e.g., "USA" -> "US")
    region_norm = normalize_region(region_raw)
    
    # Call the service function to get the structured data
    result = get_keyword_analysis(keyword, region_norm)
    
    return jsonify(result)