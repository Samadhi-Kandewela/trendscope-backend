from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import os
import json  # For handling structured data

from sqlalchemy import func, extract
from flask import current_app

from ..extensions import db
from ..models.video import Video
from ..models.clean_video import CleanVideo
from ..utils.text_utils import extract_top_keywords

# ML helpers
from ..ml.genre_forecast import get_genre_forecast_distribution
from ..ml.inference import load_trend_topic_model

# GEMINI INTEGRATION START
from google import genai
from google.genai.errors import APIError

# Global variables for the model and client instance (start as None)
_GEMINI_CLIENT_INSTANCE = None
GEMINI_MODEL = "gemini-2.5-flash"

def _get_gemini_client():
    """
    Lazily initializes the Gemini client by EXPLICITLY passing the API key 
    from os.environ, ensuring the configuration loads correctly.
    """
    global _GEMINI_CLIENT_INSTANCE
    
    if _GEMINI_CLIENT_INSTANCE is None:
        # Step 1: Explicitly retrieve the key (now guaranteed to be loaded by config.py)
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            current_app.logger.error("FATAL: GEMINI_API_KEY environment variable is missing or empty.")
            return None

        # Step 2: Pass the key directly to the client constructor
        try:
            # We explicitly pass the API key to bypass any automated system checks that failed previously.
            _GEMINI_CLIENT_INSTANCE = genai.Client(api_key=api_key)
        except Exception as e:
            current_app.logger.error("Gemini Client initialization failed: %s", str(e))
            _GEMINI_CLIENT_INSTANCE = None
    
    return _GEMINI_CLIENT_INSTANCE

# GEMINI INTEGRATION END


# ---------------------------------------------------------------------------
# Region normalization (handles "US" vs "United States" etc.)
# ---------------------------------------------------------------------------

REGION_ALIASES: Dict[str, str] = {
    # United States
    "us": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",

    # United Kingdom (YouTube uses GB, but many UIs say "UK")
    "uk": "GB",
    "united kingdom": "GB",
    "england": "GB",
    "britain": "GB",

    # India
    "india": "IN",
    "in": "IN",

    # Sri Lanka
    "sri lanka": "LK",
    "srilanka": "LK",
    "lk": "LK",

    # Added English-dominant countries (for broader filtering)
    "philippines": "PH",
    "singapore": "SG",
    "south africa": "ZA",
    "malaysia": "MY",
}


def normalize_region(region: Optional[str]) -> str:
    """
    Map user input into a canonical region identifier.
    """
    if not region:
        return "Global"
    r = region.strip()
    if not r:
        return "Global"
    key = r.lower()
    return REGION_ALIASES.get(key, r)


# ---------------------------------------------------------------------------
# 1. GENRE FORECAST (Near-Term Genres section - omitted for brevity)
# ---------------------------------------------------------------------------
# ... (Omitted code for _heuristic_genre_forecast and get_genre_forecast as they were correct)

def _heuristic_genre_forecast(region: str, days: int) -> Dict:
    """
    Fallback: simple count-based forecast using recent videos in `videos` table.
    """
    lookback_days = current_app.config.get("GENRE_FORECAST_LOOKBACK_DAYS", 60)
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    q = db.session.query(
        Video.genre.label("genre"),
        func.count(Video.id).label("cnt"),
    ).filter(Video.trending_date >= cutoff)

    if region != "Global":
        q = q.filter(Video.trending_country == region)

    rows = q.group_by("genre").all()
    total = sum(r.cnt for r in rows) or 1

    genres = [
        {"name": r.genre, "percentage": round((r.cnt / total) * 100, 1)}
        for r in rows
        if r.genre
    ]
    genres.sort(key=lambda g: g["percentage"], reverse=True)

    top = [g["name"] for g in genres[:2]]
    summary = (
        f"Based on global historical data, the most dominant content genres "
        f"for the next {days} days are predicted to be "
        + (", ".join(top) if top else "unknown")
        + ". Use the filters above to dive into specific regions and timeframes."
    )

    return {
        "region": region,
        "days": days,
        "genres": genres,
        "summary": summary,
    }


def get_genre_forecast(region: str = "Global", days: int = 30) -> Dict:
    """
    Main entry point for: GET /api/analytics/genres/near-term
    """
    region_norm = normalize_region(region)

    try:
        genres = get_genre_forecast_distribution(region_norm, days)
        top = [g["name"] for g in genres[:2]]

        summary = (
            f"Based on learned historical patterns, the most dominant content genres "
            f"for the next {days} days are predicted to be "
            f"for the next {days} days are predicted to be "
            + (", ".join(top) if top else "unknown")
            + ". Use the filters above to dive into specific regions and timeframes."
        )

        return {
            "region": region_norm,
            "days": days,
            "genres": genres,
            "summary": summary,
        }

    except Exception as e:
        current_app.logger.warning(
            "Genre forecast ML failed (%s). Falling back to heuristic.", e
        )
        return _heuristic_genre_forecast(region_norm, days)


# ---------------------------------------------------------------------------
# Helpers for seasonal trend strategy
# ---------------------------------------------------------------------------

def _months_for_range(start_date: datetime, end_date: datetime) -> List[int]:
    """
    Given an arbitrary date range (can be future), return the list of calendar
    months (1-12) to query for seasonal matching.
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    months = set()
    current = start_date.replace(day=1)

    while current.date() <= end_date.date():
        months.add(current.month)

        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

        if current.date() > end_date.date() and current.month != end_date.month:
             break

    return sorted(list(months))


def _build_text_for_clean_video(v: CleanVideo) -> str:
    """
    Build the text document for inference in the *same style* as training.
    """
    title = (v.title_clean or v.title or "").strip()
    tags = (v.tags_clean or v.tags_text or "").strip()
    desc = (v.description_clean or "").strip()

    # Truncate to match training pre-processing
    if len(desc) > 500:
        desc = desc[:500]

    return f"{title} {tags} {desc}".strip()


# GEMINI INTEGRATION START
def _generate_strategy_with_gemini(ml_signals: Dict) -> str:
    """
    Uses the Gemini API to turn structured ML signals into a polished,
    actionable strategy for the UI.
    """
    # Check if client is initialized using the lazy getter
    client = _get_gemini_client()
    if client is None:
        return (
            "AI Strategy Generator is unavailable due to configuration error. "
            "Use the marketing insights below."
        )

    # Determine positive sentiment level for prompt
    positive_ratio = ml_signals.get("sentiment_ratio", 0.0)
    sentiment_level = "highly positive" if positive_ratio > 0.95 else "strong positive"

    prompt_template = """
You are 'TrendScope', an AI content strategy expert. Your task is to analyze the provided ML signals for a specific trending topic and generate a highly detailed, professional, and actionable strategy paragraph.

The output must focus on *what* the creator should produce and *why* it will trend. Use descriptive language and reference the niche keywords.

ML Signals (JSON): {signals}

Instructions for Output:
1. Start by identifying the Core Genre and Niche Trend for the region {region} in the period {months}.
2. Focus on the top keywords: {top_keywords_str}.
3. Suggest a specific video format (e.g., 'deep dives', 'tutorials', 'unboxing', or 'review').
4. State that the audience sentiment is {sentiment_level}, implying high engagement.
5. DO NOT use the words 'based on our model' or 'predicted'. Speak with authority.
6. Ensure the output is a single, concise paragraph (max 100 words).
"""

    signals_json = json.dumps(ml_signals, indent=2)
    prompt = prompt_template.format(
        signals=signals_json,
        region=ml_signals["effectiveRegion"],
        months=ml_signals["effectiveMonths"],
        top_keywords_str=", ".join(ml_signals["topKeywords"][:5]),
        sentiment_level=sentiment_level,
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=(
                    "You are a professional content strategy expert. "
                    "Respond concisely with the requested strategy text."
                ),
            ),
        )
        return response.text.strip().replace('"', "")

    except APIError as e:
        current_app.logger.error("Gemini API call failed: %s", e)
        return (
            f"AI Service Error. Core trend: {ml_signals.get('coreTrend', 'N/A')} - "
            f"Focus on high-engagement keywords: {', '.join(ml_signals.get('topKeywords', []))}"
        )
    except Exception as e:
        current_app.logger.error("General error during Gemini call: %s", e)
        return (
            "AI service encountered an unexpected error. "
            "Use the marketing insights provided below."
        )

# GEMINI INTEGRATION END


# ---------------------------------------------------------------------------
# 2. TREND STRATEGY (Trend Strategy section)
# ---------------------------------------------------------------------------

def get_trend_strategy(
    region: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[Dict]:
    """
    Trend strategy for Analytics page. (Dynamically generates keywords and uses Gemini for text)
    """
    region_norm = normalize_region(region)
    month_list = _months_for_range(start_date, end_date)

    current_app.logger.info(
        "Trend-strategy request: region=%s (norm=%s), start=%s, end=%s, months=%s",
        region, region_norm, start_date.isoformat(), end_date.isoformat(), month_list,
    )

    # --- 1) Base query: Filter historical data by season (month) and region ---
    base_q = db.session.query(CleanVideo).filter(
        CleanVideo.trending_date.isnot(None),
        extract("month", CleanVideo.trending_date).in_(month_list),
    )

    if region_norm != "Global":
        if len(region_norm) == 2:
            base_q = base_q.filter(CleanVideo.trending_country_code == region_norm)
        else:
            base_q = base_q.filter(CleanVideo.trending_country_raw == region_norm)

    videos: List[CleanVideo] = base_q.limit(15000).all()
    if not videos:
        return None

    # --- 2) Load topic model and predict cluster for the filtered videos ---
    topic_payload = load_trend_topic_model()
    if topic_payload is None:
        current_app.logger.warning("Trend topic model not available, using fallback strategy.")
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    vectorizer = topic_payload["vectorizer"]
    kmeans = topic_payload["kmeans"]
    cluster_meta = topic_payload["cluster_meta"]

    texts: List[str] = [_build_text_for_clean_video(v) for v in videos]
    if not any(texts):
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    X = vectorizer.transform(texts)
    labels = kmeans.predict(X)

    # --- 3) Identify the MOST TRENDING Cluster for the current selection ---
    cluster_scores: Counter[int] = Counter()
    cluster_indices: Dict[int, List[int]] = defaultdict(list)

    for idx, (v, label) in enumerate(zip(videos, labels)):
        cluster_scores[label] += v.view_count or 0
        cluster_indices[label].append(idx)

    if not cluster_scores:
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    best_cluster, _ = cluster_scores.most_common(1)[0]
    meta = cluster_meta.get(best_cluster)
    if meta is None:
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    # --- 4) DYNAMIC OUTPUT GENERATION ---
    cluster_vids = [videos[i] for i in cluster_indices[best_cluster]]
    core_genre: str = meta["dominant_genre"]

    dynamic_texts: List[str] = []
    for v in cluster_vids:
        title = (v.title_clean or v.title or "")
        tags_str = (v.tags_clean or v.tags_text or "")
        dynamic_texts.append(f"{title} {tags_str}")

    top_terms: List[str] = extract_top_keywords(dynamic_texts, top_n=8)
    if not top_terms:
        top_terms = ["trending", core_genre.lower(), "viral"]

    top_cluster_vids = sorted(
        cluster_vids,
        key=lambda v: v.view_count or 0,
        reverse=True,
    )[:5]

    sample_titles: List[str] = [
        (v.title_clean or v.title or "").strip() for v in top_cluster_vids
    ]
    sample_descriptions: List[str] = [
        (v.description_clean or "").strip() for v in top_cluster_vids
    ]

    core_trend_label = (
        f"{core_genre} — {top_terms[0].title()}"
        if top_terms else core_genre
    )

    # --- 5) Sentiment proxy ---
    total_likes = 0
    total_dislikes = 0
    for v in cluster_vids:
        if v.like_count:
            total_likes += v.like_count
        if v.dislike_count:
            total_dislikes += v.dislike_count

    positive_ratio = 0.0
    if total_likes + total_dislikes > 0:
        positive_ratio = total_likes / (total_likes + total_dislikes)

    sentiment = {
        "positive": round(positive_ratio, 2),
        "neutral": round(max(0.0, (1 - positive_ratio) * 0.3), 2),
        "negative": round(max(0.0, (1 - positive_ratio) * 0.7), 2),
    }

    # --- 6) Suggested Videos ---
    suggested_videos = []
    for v in top_cluster_vids:
        vid_id = v.video_id or str(v.id)
        suggested_videos.append(
            {
                "videoId": vid_id,
                "title": v.title,
                "channel": v.channel_title or v.channel_id,
                "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/default.jpg",
                "views": v.view_count,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            }
        )

    # --- 7) Final ML signals (CONTEXT FOR GEMINI) ---
    ml_signals = {
        "coreGenre": core_genre,
        "coreTrend": core_trend_label,
        "topKeywords": top_terms,
        "representativeTitles": sample_titles,
        "representativeDescriptions": sample_descriptions,
        "effectiveRegion": region_norm,
        "effectiveMonths": month_list,
        "sentiment_ratio": positive_ratio,
    }

    # --- 8) GEMINI API CALL REPLACES STATIC STRING ---
    detailed_strategy = _generate_strategy_with_gemini(ml_signals)

    return {
        "region": region_norm,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "coreTrend": core_trend_label,
        "coreGenre": core_genre,
        "detailedStrategy": detailed_strategy,
        "sentiment": sentiment,
        "suggestedVideos": suggested_videos,
        "marketingInsights": top_terms[:5],
        "mlSignals": ml_signals,
        "strategySource": "ml-topic-model-gemini",
    }


# ---------------------------------------------------------------------------
# 3. Fallback trend strategy (no topic model available)
# ---------------------------------------------------------------------------
# ... (Omitted code for _fallback_trend_strategy as it was correct)
def _fallback_trend_strategy(
    region: str,
    start_date: datetime,
    end_date: datetime,
    videos: List[CleanVideo | Video],
) -> Optional[Dict]:
    """
    Simple heuristic version used only when the topic model is unavailable.
    """
    if not videos:
        return None

    # 1) Pick dominant genre
    genre_counts = Counter(v.genre for v in videos if v.genre)
    core_genre, _ = genre_counts.most_common(1)[0]
    genre_videos = [v for v in videos if v.genre == core_genre]

    # 2) Extract top keywords from titles + tags
    texts: List[str] = []
    for v in genre_videos:
        title = getattr(v, "title_clean", None) or v.title or ""
        tags_str = getattr(v, "tags_clean", None) or " ".join(v.tags or [])
        texts.append(f"{title} {tags_str}")

    top_keywords = extract_top_keywords(texts, top_n=8)
    core_trend_label = (
        f"{core_genre} — {top_keywords[0].title()}"
        if top_keywords else core_genre
    )

    # Fallback uses the old static string description
    detailed_strategy = (
        f"For the selected period, {core_genre} content is expected to perform strongly in {region}. "
        f"Creators should focus on topics like: {', '.join(top_keywords[:5])}. "
        f"Consider comparison videos, hands-on demos, and step-by-step guides using these themes. "
        f"Optimize thumbnails and titles with 1–2 of the keywords below for maximum click-through."
    )

    # 3) Sentiment proxy
    total_likes = 0
    total_dislikes = 0
    for v in genre_videos:
        if v.like_count:
            total_likes += v.like_count
        if v.dislike_count:
            total_dislikes += v.dislike_count

    positive_ratio = 0.0
    if total_likes + total_dislikes > 0:
        positive_ratio = total_likes / (total_likes + total_dislikes)

    sentiment = {
        "positive": round(positive_ratio, 2),
        "neutral": round(max(0.0, 1 - positive_ratio) * 0.3, 2),
        "negative": round(max(0.0, 1 - positive_ratio) * 0.7, 2),
    }

    # 4) Suggested videos (top 5 by views in this genre)
    top_videos = sorted(
        genre_videos,
        key=lambda v: v.view_count or 0,
        reverse=True,
    )[:5]

    suggested_videos = []
    for v in top_videos:
        vid_id = getattr(v, "video_id", None) or v.id
        thumbnail = f"https://i.ytimg.com/vi/{vid_id}/default.jpg"
        suggested_videos.append(
            {
                "videoId": vid_id,
                "title": v.title,
                "channel": v.channel_title or v.channel_id,
                "thumbnail": thumbnail,
                "views": v.view_count,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            }
        )

    marketing_insights = top_keywords[:5]

    return {
        "region": region,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "coreTrend": core_trend_label,
        "coreGenre": core_genre,
        "detailedStrategy": detailed_strategy,
        "sentiment": sentiment,
        "suggestedVideos": suggested_videos,
        "marketingInsights": marketing_insights,
        "mlSignals": {
            "coreGenre": core_genre,
            "topKeywords": top_keywords,
            "sentiment_ratio": positive_ratio,
        },
        "strategySource": "heuristic-fallback",
    }