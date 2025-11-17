from flask import Blueprint, request, jsonify
from ..services.youtube_service import get_trending_videos

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.get("/trending-videos")
def trending_videos():
    """
    GET /api/dashboard/trending-videos?genre=Gaming&region=US&limit=8
    Returns a list of current trending YouTube videos for the given genre & region.
    """
    genre = request.args.get("genre", "Gaming")
    region = request.args.get("region", "US")
    limit = int(request.args.get("limit", 8))

    videos = get_trending_videos(region=region, genre=genre, limit=limit)
    return jsonify({
        "genre": genre,
        "region": region,
        "count": len(videos),
        "videos": videos,
    })
