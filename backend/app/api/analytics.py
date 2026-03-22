from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
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
    model_type = data.get("modelType", "legacy")

    # If useAdvanced is true but modelType is not set, default to 'advanced'
    if use_advanced and model_type == "legacy":
        model_type = "advanced"

    # Extract user_id from JWT if authenticated (for personalized strategies)
    user_id = None
    try:
        from ..utils.auth import decode_token
        # Check header first, then query param as fallback (for Postman)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            auth_header = request.args.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            payload = decode_token(token)
            if payload:
                user_id = payload.get("user_id") or payload.get("sub")
                current_app.logger.info("Authenticated user_id=%s for trend-strategy", user_id)
    except Exception as e:
        current_app.logger.debug("Auth extraction failed: %s", e)
        pass  # Not authenticated — use generic strategy

    result = get_trend_strategy(
        region=region,
        start_date=start_date,
        end_date=end_date,
        use_advanced_model=use_advanced,
        model_type=model_type,
        user_id=user_id,
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


@analytics_bp.post("/title-optimizer")
def optimize_title():
    """
    POST /api/analytics/title-optimizer
    Body: {
      "draftTitle": "My Trip to Japan",
      "genre": "Travel",         (optional — auto-detected from profile if authed)
      "region": "US"             (optional, defaults to "US")
    }
    """
    from ..utils.text_utils import extract_top_keywords
    from ..services.youtube_service import _get_gemini_client
    from ..models.clean_video import CleanVideo
    from ..extensions import db
    import json
    from google import genai

    data = request.get_json() or {}
    draft_title = (data.get("draftTitle") or "").strip()
    genre = data.get("genre")
    region = data.get("region", "US")

    if not draft_title:
        return jsonify({"error": "draftTitle is required"}), 400

    # Auto-detect genre from profile if authenticated
    if not genre:
        try:
            from ..utils.auth import decode_token
            from ..models.creator_profile import CreatorProfile
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]
                payload = decode_token(token)
                if payload:
                    uid = payload.get("user_id") or payload.get("sub")
                    if uid:
                        profile = CreatorProfile.query.filter_by(user_id=uid).first()
                        if profile:
                            genre = profile.primary_genre
        except Exception:
            pass

    if not genre:
        genre = "Lifestyle"  # Default

    # Get trending keywords from DB for this genre
    try:
        from sqlalchemy import extract
        from datetime import datetime
        now = datetime.utcnow()
        recent_videos = CleanVideo.query.filter(
            CleanVideo.genre == genre,
        ).order_by(CleanVideo.view_count.desc()).limit(200).all()

        texts = [
            f"{v.title_clean or v.title or ''} {v.tags_clean or ''}"
            for v in recent_videos
        ]
        trending_keywords = extract_top_keywords(texts, top_n=15) if texts else []
    except Exception:
        trending_keywords = []

    # Use Gemini to optimize
    client = _get_gemini_client()
    if not client:
        return jsonify({"error": "AI service unavailable"}), 503

    prompt = f"""
You are a YouTube SEO expert. Optimize this video title for maximum CTR and discoverability.

DRAFT TITLE: "{draft_title}"
CREATOR'S GENRE: {genre}
REGION: {region}
TRENDING KEYWORDS IN {genre}: {', '.join(trending_keywords[:10])}

Generate 3 optimized title alternatives. For each, explain why it's better.

OUTPUT FORMAT (JSON ONLY):
{{
  "originalTitle": "{draft_title}",
  "suggestions": [
    {{
      "optimizedTitle": "The optimized title",
      "whyBetter": "Which trending keywords were added and why they boost CTR",
      "trendingKeywordsUsed": ["keyword1", "keyword2"]
    }}
  ],
  "trendingKeywordsInNiche": {json.dumps(trending_keywords[:10])},
  "tips": ["General tip 1 for titles in this niche", "Tip 2"]
}}

RULES:
- JSON only. No markdown.
- Titles should be 50-70 characters, engaging, include numbers or brackets where appropriate.
- Each suggestion should use at least 1-2 trending keywords naturally.
- Tips should be specific to {genre} content.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        result = json.loads(response.text.strip())
        result["genre"] = genre
        result["region"] = region
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error("Title optimizer failed: %s", e)
        return jsonify({"error": str(e)}), 500
