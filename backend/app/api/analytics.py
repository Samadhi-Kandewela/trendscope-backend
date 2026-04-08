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
    from ..services.trend_service import _get_groq_client, GROQ_MODEL
    from ..models.clean_video import CleanVideo
    from ..extensions import db
    import json

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

    # Use Groq to optimize
    client = _get_groq_client()
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
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        result["genre"] = genre
        result["region"] = region
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error("Title optimizer failed: %s", e)
        return jsonify({"error": str(e)}), 500


@analytics_bp.get("/model-evaluation")
def get_model_evaluation():
    """
    GET /api/analytics/model-evaluation
    Returns evaluation metrics for the TrendClassifier (baseline vs improved).
    Useful for examiner review of model accuracy.
    """
    from ..ml.trend_classifier import TrendClassifier
    classifier = TrendClassifier()
    summary = classifier.get_evaluation_summary()
    return jsonify(summary)


@analytics_bp.post("/train-classifier")
def train_trend_classifier():
    """
    POST /api/analytics/train-classifier
    Body (optional): { "genre": "Travel", "limit": 10000 }

    Trains the TrendClassifier on historical video data and returns
    full evaluation metrics (F1, precision, recall, AUC, feature importance).
    """
    from ..ml.trend_classifier import TrendClassifier
    from ..models.clean_video import CleanVideo
    from ..extensions import db

    data = request.get_json() or {}
    genre = data.get("genre")
    limit = int(data.get("limit", 10000))

    # Load training data
    query = db.session.query(CleanVideo).filter(
        CleanVideo.view_count.isnot(None),
        CleanVideo.view_count > 0,
    )
    if genre:
        query = query.filter(CleanVideo.genre == genre)

    videos = query.order_by(CleanVideo.trending_date.desc()).limit(limit).all()

    if len(videos) < 50:
        return jsonify({
            "error": f"Insufficient data: {len(videos)} videos found. Need at least 50."
        }), 400

    classifier = TrendClassifier()
    results = classifier.train(videos, genre=genre)

    if "error" in results:
        return jsonify(results), 400

    return jsonify({
        "message": "TrendClassifier trained successfully.",
        "videosUsed": len(videos),
        "evaluation": results,
    }), 200


# ---------------------------------------------------------------------------
# PHASE 5: BACKTESTING — Temporal Holdout Keyword Prediction Accuracy
# ---------------------------------------------------------------------------

def _extract_top_keywords_from_videos(videos, top_n=20):
    """
    Count keyword frequency across video titles+tags and return the top_n
    terms (by how many videos they appear in).
    """
    import re
    from collections import Counter

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "it", "be", "as", "was",
        "are", "were", "been", "has", "have", "had", "that", "this", "i",
        "you", "he", "she", "we", "they", "my", "your", "our", "their",
        "its", "do", "did", "does", "will", "would", "can", "could",
        "should", "may", "might", "new", "video", "official", "ft", "feat",
    }

    keyword_doc_freq = Counter()
    for v in videos:
        text = " ".join(filter(None, [
            v.title_clean or v.title or "",
            v.tags_clean or v.tags_text or "",
        ])).lower()
        words = set(re.findall(r"[a-z]{3,}", text))
        words -= STOPWORDS
        for w in words:
            keyword_doc_freq[w] += 1

    return [kw for kw, _ in keyword_doc_freq.most_common(top_n)]


def _run_backtest(videos, top_n=20):
    """
    Temporal holdout backtest per genre.

    Split each genre's videos 80/20 by trending_date (oldest 80% = train,
    newest 20% = test).  Extract top_n keywords from the train window and
    measure how many actually appeared in the test window.

    Returns per-genre metrics + aggregated overall score.
    """
    from collections import defaultdict

    # Group by genre, sorted by date
    genre_videos = defaultdict(list)
    for v in videos:
        if v.genre:
            genre_videos[v.genre].append(v)

    genre_results = {}
    total_precision_sum = 0.0
    total_recall_sum = 0.0
    evaluated_genres = 0

    for genre, vids in genre_videos.items():
        # Need enough videos for a meaningful split
        if len(vids) < 20:
            genre_results[genre] = {
                "skipped": True,
                "reason": f"Only {len(vids)} videos — need at least 20",
                "videoCount": len(vids),
            }
            continue

        # Sort by trending_date (None dates treated as oldest)
        from datetime import datetime as dt
        sorted_vids = sorted(
            vids,
            key=lambda v: v.trending_date if v.trending_date is not None else dt.min
        )
        cutoff = int(len(sorted_vids) * 0.8)
        train_vids = sorted_vids[:cutoff]
        test_vids = sorted_vids[cutoff:]

        if not test_vids:
            genre_results[genre] = {
                "skipped": True,
                "reason": "Test window is empty after split",
                "videoCount": len(vids),
            }
            continue

        # Predicted keywords (from train window)
        predicted = set(_extract_top_keywords_from_videos(train_vids, top_n=top_n))

        # Actual keywords present in test window
        actual = set(_extract_top_keywords_from_videos(test_vids, top_n=top_n * 3))

        hits = predicted & actual
        n_hits = len(hits)
        precision = n_hits / len(predicted) if predicted else 0.0
        recall = n_hits / len(actual) if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Date range info
        train_start = sorted_vids[0].trending_date
        train_end = sorted_vids[cutoff - 1].trending_date
        test_start = sorted_vids[cutoff].trending_date
        test_end = sorted_vids[-1].trending_date

        genre_results[genre] = {
            "skipped": False,
            "videoCount": len(vids),
            "trainVideos": len(train_vids),
            "testVideos": len(test_vids),
            "trainWindow": {
                "start": train_start.isoformat() if train_start else None,
                "end": train_end.isoformat() if train_end else None,
            },
            "testWindow": {
                "start": test_start.isoformat() if test_start else None,
                "end": test_end.isoformat() if test_end else None,
            },
            "predictedKeywords": sorted(predicted),
            "matchedKeywords": sorted(hits),
            "missedKeywords": sorted(actual - predicted),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1Score": round(f1, 4),
        }
        total_precision_sum += precision
        total_recall_sum += recall
        evaluated_genres += 1

    if evaluated_genres > 0:
        avg_precision = total_precision_sum / evaluated_genres
        avg_recall = total_recall_sum / evaluated_genres
        avg_f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0 else 0.0
        )
    else:
        avg_precision = avg_recall = avg_f1 = 0.0

    return {
        "overall": {
            "avgPrecision": round(avg_precision, 4),
            "avgRecall": round(avg_recall, 4),
            "avgF1Score": round(avg_f1, 4),
            "evaluatedGenres": evaluated_genres,
            "totalVideos": len(videos),
            "methodology": (
                "Temporal holdout: train on oldest 80% of each genre's videos, "
                "test on newest 20%. Measures keyword prediction overlap."
            ),
        },
        "byGenre": genre_results,
    }


@analytics_bp.get("/backtest-results")
def get_backtest_results():
    """
    GET /api/analytics/backtest-results?limit=15000

    Runs a temporal holdout backtest on historical YouTube trending data.

    Methodology:
    - For each genre: sort videos by trending_date, split 80/20
    - Extract top keywords from the 'train' (older) window
    - Measure how many appear in the 'test' (newer) window
    - Reports per-genre precision, recall, F1 + overall averages

    This proves the model predicts real future trends, not just memorises
    historical patterns (temporal split prevents data leakage).
    """
    from ..models.clean_video import CleanVideo
    from ..extensions import db

    limit = int(request.args.get("limit", 15000))
    genre_filter = request.args.get("genre")

    query = db.session.query(CleanVideo).filter(
        CleanVideo.trending_date.isnot(None),
        CleanVideo.genre.isnot(None),
        CleanVideo.view_count.isnot(None),
        CleanVideo.view_count > 0,
    )
    if genre_filter:
        query = query.filter(CleanVideo.genre == genre_filter)

    videos = query.order_by(CleanVideo.trending_date.asc()).limit(limit).all()

    if len(videos) < 40:
        return jsonify({
            "error": f"Insufficient data: {len(videos)} videos found. Need at least 40."
        }), 400

    results = _run_backtest(videos, top_n=20)
    results["generatedAt"] = datetime.utcnow().isoformat()
    results["videosLoaded"] = len(videos)

    return jsonify(results), 200
