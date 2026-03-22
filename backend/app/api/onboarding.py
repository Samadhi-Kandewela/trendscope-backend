# backend/app/api/onboarding.py
"""
Creator Onboarding API.

Endpoints:
  POST /api/onboarding/profile   – Save / update creator profile
  GET  /api/onboarding/profile   – Get current user's creator profile
  POST /api/onboarding/fetch-channel – Fetch YouTube channel stats
"""
from datetime import datetime
from flask import Blueprint, request, jsonify

from ..extensions import db
from ..models.creator_profile import CreatorProfile
from ..utils.auth import login_required

onboarding_bp = Blueprint("onboarding", __name__)

# Valid enum values for dropdowns
VALID_GENRES = [
    "Gaming", "Music", "Entertainment", "Education", "Travel",
    "Beauty", "Tech", "Food", "Sports", "News", "Comedy",
    "Science", "Health", "Finance", "Lifestyle", "Other",
]
VALID_STYLES = [
    "Vlogs", "Tutorials", "Reviews", "Shorts", "Podcasts",
    "Live Streams", "Documentary", "Commentary", "Interviews", "Other",
]
VALID_AGE_RANGES = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
VALID_GOALS = [
    "grow_subscribers", "increase_views", "brand_deals",
    "community_building", "education", "monetization",
]


@onboarding_bp.post("/profile")
@login_required
def save_profile():
    """
    POST /api/onboarding/profile
    Body: {
      "channelUrl": "https://www.youtube.com/@TravelWithMe",
      "primaryGenre": "Travel",
      "contentStyle": "Vlogs",
      "targetAudienceAge": "18-24",
      "targetRegion": "US",
      "creatorGoal": "grow_subscribers"
    }
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}

    # Look up or create
    profile = CreatorProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        profile = CreatorProfile(user_id=user.id)
        db.session.add(profile)

    # Update fields (only if provided)
    if "channelUrl" in data:
        profile.channel_url = (data["channelUrl"] or "").strip()
    if "primaryGenre" in data:
        genre = data["primaryGenre"]
        if genre and genre not in VALID_GENRES:
            return jsonify({"error": f"Invalid genre. Must be one of: {VALID_GENRES}"}), 400
        profile.primary_genre = genre
    if "contentStyle" in data:
        style = data["contentStyle"]
        if style and style not in VALID_STYLES:
            return jsonify({"error": f"Invalid style. Must be one of: {VALID_STYLES}"}), 400
        profile.content_style = style
    if "targetAudienceAge" in data:
        age = data["targetAudienceAge"]
        if age and age not in VALID_AGE_RANGES:
            return jsonify({"error": f"Invalid age range. Must be one of: {VALID_AGE_RANGES}"}), 400
        profile.target_audience_age = age
    if "targetRegion" in data:
        profile.target_region = (data["targetRegion"] or "").strip().upper() or "US"
    if "creatorGoal" in data:
        goal = data["creatorGoal"]
        if goal and goal not in VALID_GOALS:
            return jsonify({"error": f"Invalid goal. Must be one of: {VALID_GOALS}"}), 400
        profile.creator_goal = goal

    # Mark onboarding completed if all core fields are filled
    core_filled = all([
        profile.primary_genre,
        profile.content_style,
        profile.target_audience_age,
        profile.creator_goal,
    ])
    profile.onboarding_completed = core_filled

    db.session.commit()

    return jsonify({"profile": profile.to_dict()}), 201


@onboarding_bp.get("/profile")
@login_required
def get_profile():
    """
    GET /api/onboarding/profile
    Returns the current user's creator profile (or 404).
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    profile = CreatorProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        return jsonify({"profile": None, "onboardingCompleted": False}), 200

    return jsonify({"profile": profile.to_dict(), "onboardingCompleted": profile.onboarding_completed}), 200


@onboarding_bp.post("/fetch-channel")
@login_required
def fetch_channel():
    """
    POST /api/onboarding/fetch-channel
    Body: { "channelUrl": "https://www.youtube.com/@TravelWithMe" }
    Fetches YouTube channel stats and saves them to the creator profile.
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    channel_url = (data.get("channelUrl") or "").strip()

    if not channel_url:
        return jsonify({"error": "channelUrl is required"}), 400

    # Fetch channel info via YouTube API
    from ..services.youtube_service import fetch_channel_info
    channel_data = fetch_channel_info(channel_url)

    if not channel_data or channel_data.get("error"):
        return jsonify({
            "error": channel_data.get("error", "Could not fetch channel data")
        }), 400

    # Save to profile
    profile = CreatorProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        profile = CreatorProfile(user_id=user.id)
        db.session.add(profile)

    profile.channel_url = channel_url
    profile.channel_id = channel_data.get("channelId")
    profile.channel_name = channel_data.get("channelName")
    profile.subscriber_count = channel_data.get("subscriberCount")
    profile.total_views = channel_data.get("totalViews")
    profile.video_count = channel_data.get("videoCount")
    profile.channel_description = channel_data.get("description")
    profile.channel_thumbnail = channel_data.get("thumbnail")
    profile.channel_data_fetched_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        "channel": channel_data,
        "profile": profile.to_dict(),
    }), 200


@onboarding_bp.get("/enums")
def get_enums():
    """
    GET /api/onboarding/enums
    Returns valid dropdown values for the onboarding form.
    No auth required — frontend needs these before login.
    """
    return jsonify({
        "genres": VALID_GENRES,
        "contentStyles": VALID_STYLES,
        "ageRanges": VALID_AGE_RANGES,
        "goals": [
            {"value": "grow_subscribers", "label": "Grow Subscribers"},
            {"value": "increase_views", "label": "Increase Views"},
            {"value": "brand_deals", "label": "Get Brand Deals"},
            {"value": "community_building", "label": "Build Community"},
            {"value": "education", "label": "Educate Audience"},
            {"value": "monetization", "label": "Monetize Content"},
        ],
    }), 200
