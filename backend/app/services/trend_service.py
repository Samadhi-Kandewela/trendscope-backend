from datetime import datetime, timedelta, date
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from calendar import month_name

import os
print("DEBUG: LOADED NEW TREND SERVICE (Step 5149)")
import json  # For handling structured data
from textblob import TextBlob  # NEW: For real sentiment analysis
from sklearn.metrics.pairwise import cosine_similarity # NEW: For video similarity
import numpy as np

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
    "malaysia": "MY",
}
from ..models.clean_video import CleanVideo # Added this import to avoid circular dependency
from ..ml.inference import MODELS_DIR
from ..ml.lightweight_inference import HybridInferenceEngine
import os

# Global Cache for Inference Engine (loaded once)
_hybrid_engine = None

def _get_hybrid_engine():
    global _hybrid_engine
    json_path = MODELS_DIR / "hybrid_model_params.json"
    
    if _hybrid_engine is None:
        if json_path.exists():
            current_app.logger.info("Loading lightweight hybrid engine from %s...", json_path)
            _hybrid_engine = HybridInferenceEngine(json_path)
        else:
            current_app.logger.warning("Hybrid JSON model missing: %s", json_path)
    return _hybrid_engine

def _train_on_the_fly_hybrid():
    """
    Emergency training function. If the hybrid model PKL is missing, 
    train it right now using a small sample of recent videos.
    """
    current_app.logger.warning("Training HYBRID MODEL (TF-IDF + K-Means) on-the-fly...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import joblib
        from pathlib import Path

        # Fetch limited sample (FAST)
        data = db.session.query(CleanVideo).order_by(CleanVideo.trending_date.desc()).limit(1000).all()
        if not data:
            current_app.logger.error("No data available for training.")
            return None
        
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        texts = [_build_text_for_clean_video(v) for v in data]
        embeddings = vectorizer.fit_transform(texts)
        
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=5) # Reduced complexity
        labels = kmeans.fit_predict(embeddings)
        
        cluster_meta = {}
        # Simple metadata generation (skipped extensive logic for speed)
        
        payload = {
            "vectorizer": vectorizer,
            "kmeans_model": kmeans,
            "cluster_meta": cluster_meta
        }
        
        # Save it
        path = MODELS_DIR / "hybrid_trend_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        
        current_app.logger.info("Hybrid model trained and saved to %s", path)
        return payload

    except Exception as e:
        current_app.logger.error("On-the-fly training failed: %s", e)
        return None


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


def get_genre_forecast(
    region: str = "Global", 
    days: int = 30,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict:
    """
    Main entry point for: GET /api/analytics/genres/near-term
    """
    region_norm = normalize_region(region)

    try:
        # Pass explicit dates if available, otherwise days will be handled by default logic
        genres = get_genre_forecast_distribution(region_norm, start_date=start_date, end_date=end_date)

        # Build a clean human-readable summary
        names = [g["name"] for g in genres]

        top1 = names[0] if len(names) > 0 else None
        top2 = names[1] if len(names) > 1 else None
        top3 = names[2] if len(names) > 2 else None
        top4 = names[3] if len(names) > 3 else None

        parts = []
        
        # Determine time description
        if start_date and end_date:
            s_str = start_date.strftime("%b %d")
            e_str = end_date.strftime("%b %d")
            time_desc = f"between {s_str} and {e_str}"
        else:
            time_desc = f"over the next {days} days"

        # Main statement – top 1–2 genres
        if top1 and top2:
            parts.append(
                f"The most dominant content genres expected {time_desc} in {region_norm} are {top1} and {top2}"
            )
        elif top1:
            parts.append(
                f"The leading content genre for {time_desc} in {region_norm} is expected to be {top1}"
            )
        else:
            parts.append(
                f"No strong dominant genres were detected for {time_desc} in {region_norm}"
            )

        # Optional “followed by …” for the next 1–2 genres
        follow = []
        if top3:
            follow.append(top3)
        if top4:
            follow.append(top4)
        if follow:
            parts.append("followed by " + ", ".join(follow))

        # Closing sentence (Removed as per user request for conciseness)


        summary = ". ".join(parts) + "."

        if start_date and end_date:
            days = abs((end_date - start_date).days)
            if days == 0:
                days = 1 # Avoid 0-day text

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

def _build_strategy_from_signals(ml_signals: Dict) -> str:
    """
    Fallback: build a creator-friendly strategy paragraph using only ML signals,
    shown when Gemini is unavailable or fails.
    """
    # Mark source as ML fallback
    ml_signals["strategyEngine"] = "ml-fallback"

    core_genre = ml_signals.get("coreGenre") or "your main niche"
    core_trend = ml_signals.get("coreTrend") or core_genre
    top_keywords: List[str] = ml_signals.get("topKeywords") or []
    region = ml_signals.get("effectiveRegion") or "your target region"
    months = ml_signals.get("effectiveMonths") or []

    # Keywords text
    if top_keywords:
        primary_kw = top_keywords[0]
        other_kws = ", ".join(top_keywords[1:4])
        if other_kws:
            keywords_str = f"{primary_kw} (plus related terms like {other_kws})"
        else:
            keywords_str = primary_kw
    else:
        keywords_str = "the most searched terms in your niche"

    # Period text
    if months:
        month_labels = ", ".join(
            month_name[m] for m in months if isinstance(m, int) and 1 <= m <= 12
        )
        period_str = f"during the upcoming {len(months)} month(s) ({month_labels})"
    else:
        period_str = "over the upcoming weeks"

    # Sentiment description
    sentiment_ratio = float(ml_signals.get("sentiment_ratio", 0.0) or 0.0)
    if sentiment_ratio >= 0.9:
        sentiment_desc = "viewers are extremely receptive and likely to binge related videos"
    elif sentiment_ratio >= 0.8:
        sentiment_desc = "audience interest is very strong with good potential for high watch time"
    elif sentiment_ratio >= 0.7:
        sentiment_desc = "there is a solid, growing interest that rewards clear value and strong hooks"
    else:
        sentiment_desc = "interest exists but viewers are selective, so clarity and value are crucial"

    return (
        f"For creators targeting {region}, the most promising opportunity right now sits inside "
        f"the {core_genre} space, with a strong pull around the trend “{core_trend}”. "
        f"Focus your next videos on practical, search-friendly topics built around {keywords_str}. "
        f"Combine formats like step-by-step tutorials, comparison or review videos, and short explainer clips "
        f"that answer very specific questions viewers have {period_str}. "
        f"Use clean, benefit-driven titles that highlight the main keyword and the transformation a viewer gets, "
        f"and pair them with thumbnails that show a clear before-and-after or bold result. "
        f"Overall {sentiment_desc}, so lean into consistent uploads and consider turning this into a small series "
        f"to keep viewers coming back for related content."
    )


# GEMINI INTEGRATION START
def _generate_strategy_with_gemini(ml_signals: Dict) -> Dict:
    """
    Uses the Gemini API to turn structured ML signals into a STRUCTURED,
    personalized strategy with actionable bullet points.
    """
    client = _get_gemini_client()
    if client is None:
        return {"detailedStrategy": _build_strategy_from_signals(ml_signals), "marketingHooks": []}

    positive_ratio = ml_signals.get("sentiment_ratio", 0.0)
    sentiment_level = "highly positive" if positive_ratio > 0.95 else "strong positive"

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    current_year = today.year

    # ── Check if personalized ──
    creator_profile = ml_signals.get("creatorProfile")
    is_personalized = bool(creator_profile)

    if is_personalized:
        genre = creator_profile.get('primaryGenre', 'general')
        style = creator_profile.get('contentStyle', 'mixed')
        subs = creator_profile.get('subscriberCount', 0)
        goal = creator_profile.get('creatorGoal', 'growth')
        audience_age = creator_profile.get('targetAudienceAge', 'all ages')
        audience_region = creator_profile.get('targetRegion', 'Global')

        prompt_template = """
You are 'TrendScope', a smart AI content advisor speaking DIRECTLY to a YouTube creator on their personal dashboard.

Current date: {today_str}
Context: {region}, months {months}
Trending Signals from ML analysis: {signals}

CREATOR PROFILE:
- Genre: {genre}
- Content Style: {style}
- Subscribers: {subs}
- Goal: {goal}
- Target Audience: {audience_age} in {audience_region}

CRITICAL RULES:
1. NEVER mention the creator's name or channel name. Address them as "you/your".
2. This is THEIR dashboard — speak directly, casually, like a smart advisor friend.
3. Cross-reference trending keywords with THEIR genre creatively.
   Example: If "Food" is trending and they do Travel, suggest "Street Food Tours", "Culinary Destination Guides".
4. Be specific with data signals (use +XX%, "rising keyword", "crossing over from X").
5. Content gaps should feel like insider advice: "You haven't covered X, but it's the #N rising keyword..."
6. Title suggestions must show a BAD generic title vs YOUR optimized version with trending keywords.

OUTPUT FORMAT (JSON ONLY):
{{
  "personalizedIntro": "1-2 sentence punchy intro with emoji. Example: '🎯 Here\\'s your personalized strategy for March 2026 — 3 emerging opportunities match your niche perfectly.'",
  "emergingTrends": [
    {{
      "trend": "Name of the trend",
      "signal": "Short data signal, e.g. '+45% search volume' or 'crossing over from Lifestyle into Travel'",
      "relevanceToYou": "Why this matters for YOUR genre. 1 sentence, speak directly."
    }}
  ],
  "recommendedAngles": [
    {{
      "angle": "A specific, creative content idea/angle",
      "why": "1 sentence: which trending keywords this leverages"
    }}
  ],
  "contentGaps": [
    {{
      "topic": "A trending topic you should cover but likely haven't",
      "insight": "Data-backed reason, e.g. 'It\\'s the #2 rising keyword in your niche. Creators covering this got 2.5x more views.'"
    }}
  ],
  "titleSuggestions": [
    {{
      "draft": "A generic title a creator might use (bad example)",
      "optimized": "The SEO-optimized version with trending keywords baked in",
      "whyBetter": "Which trending keywords were added and why they boost CTR"
    }}
  ],
  "optimalPosting": {{
    "bestDays": "e.g. 'Tuesdays & Thursdays'",
    "bestTime": "e.g. '2pm - 5pm EST'",
    "reason": "Brief reason based on audience age + region demographics"
  }},
  "marketingHooks": ["Hook 1", "Hook 2", "Hook 3", "Hook 4", "Hook 5"],
  "detailedStrategy": "A concise 2-3 sentence summary tying everything together. Speak directly to the creator."
}}

RULES:
- JSON only. No markdown. No code fences.
- emergingTrends: exactly 3 items.
- recommendedAngles: exactly 3 items. Be creatively specific, not generic.
- contentGaps: exactly 2 items. Reference trending keywords related to their genre.
- titleSuggestions: exactly 2 examples.
- marketingHooks: exactly 5 punchy multi-word phrases.
- optimalPosting: Base on their audience age ({audience_age}) and region ({audience_region}).
"""
        prompt_vars = dict(
            signals=json.dumps(
                {k: v for k, v in ml_signals.items() if k != "creatorProfile"},
                indent=2,
            ),
            region=ml_signals["effectiveRegion"],
            months=ml_signals["effectiveMonths"],
            today_str=today_str,
            current_year=current_year,
            genre=genre,
            style=style,
            subs=subs,
            goal=goal,
            audience_age=audience_age,
            audience_region=audience_region,
        )
    else:
        # ── Generic (non-personalized) prompt ──
        prompt_template = """
You are 'TrendScope', an AI trend analysis expert.
Analyze HISTORICAL DATA to PREDICT a strategy for {current_year}.

Current date: {today_str}
Context: {region}, {months}
Signals: {signals}

GOAL:
1. Predict the MODERN trend for {current_year}.
2. Generate 5 "Marketing Hooks" (buzzwords/phrases).

OUTPUT FORMAT (JSON ONLY):
{{
  "detailedStrategy": "A single rich paragraph (130-180 words) focusing on the opportunity, content strategy, and predicted sentiment.",
  "marketingHooks": ["Hook 1", "Hook 2", "Hook 3", "Hook 4", "Hook 5"]
}}

RULES:
- JSON only. No markdown.
- Hooks should be punchy (e.g., "Cozy Gaming Setup", "ASMR Study", not just "gaming").
"""
        prompt_vars = dict(
            signals=json.dumps(
                {k: v for k, v in ml_signals.items() if k != "creatorProfile"},
                indent=2,
            ),
            region=ml_signals["effectiveRegion"],
            months=ml_signals["effectiveMonths"],
            sentiment_level=sentiment_level,
            today_str=today_str,
            current_year=current_year,
        )

    prompt = prompt_template.format(**prompt_vars)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        text = response.text.strip()
        result = json.loads(text)

        ml_signals["strategyEngine"] = "gemini"
        ml_signals["isPersonalized"] = is_personalized
        return result

    except Exception as e:
        current_app.logger.error("Gemini API call failed: %s", e)
        return {"detailedStrategy": _build_strategy_from_signals(ml_signals), "marketingHooks": []}

# GEMINI INTEGRATION END

from scipy.sparse import vstack # NEW: For local centroid calculation
from sqlalchemy import func, extract
from flask import current_app

# ... imports ...


# ---------------------------------------------------------------------------
# 2. TREND STRATEGY (Trend Strategy section)
# ---------------------------------------------------------------------------

def _calculate_trend_metrics_normalized(keywords: List[str], cluster_vids: List['CleanVideo']) -> Dict:
    """
    Computes real engagement metrics for the top keywords in the cluster using 0-100 normalized scores.
    Returns: { keyword: { volume, avgViews, frequency, growth, trendScore } }
    """
    metrics = {}
    n_vids = len(cluster_vids)
    if n_vids == 0: 
        return {}

    for kw in keywords:
        kw_lower = kw.lower()
        # Find videos containing this keyword
        match_vids = [
            v for v in cluster_vids 
            if kw_lower in (v.title_clean or "").lower() 
            or kw_lower in (v.description_clean or "").lower()
        ]
        count = len(match_vids)
        
        if count > 0:
            kw_views = sum(v.view_count or 0 for v in match_vids)
            avg_views = int(kw_views / count)
            freq = round((count / n_vids) * 100, 1)
        else:
            avg_views = 0
            freq = 0.0
            
        # Normalization Logic (0-100)
        # Volume Score: 10% frequency is "High" (100)
        vol_score = min(100.0, freq * 10.0) 
        
        # View Score: 500k avg views is "High" (100)
        view_score = min(100.0, (avg_views / 500000.0) * 100.0)
        
        # Weighted Combination
        raw_score = (0.4 * vol_score) + (0.6 * view_score)
        final_score = int(min(100, max(0, raw_score)))

        metrics[kw] = {
            "volume": count,
            "avgViews": avg_views,
            "frequency": freq, # % of cluster videos containing this keyword
            "growth": 0.0, # Placeholder (requires historical DB query)
            "trendScore": final_score
        }
    return metrics

def get_trend_strategy(
    region: str,
    start_date: datetime,
    end_date: datetime,
    use_advanced_model: bool = False,
    model_type: str = "legacy", # Options: "legacy", "advanced", "hybrid"
    user_id: Optional[int] = None,  # NEW: For personalized strategies
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

    # GUARD: If model_type is hybrid, force disable S-BERT (known to crash on Windows)
    if model_type == "hybrid":
        use_advanced_model = False

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

    # --- 1b) Fetch LIVE API videos (High Priority) ---
    from ..models.video import Video
    live_q = db.session.query(Video).filter(
        Video.source_dataset == 'live_api'
    )
    # Filter live videos by region if not Global
    if region_norm != "Global":
        # Video.trending_country is usually the code (US, IN)
        live_q = live_q.filter(Video.trending_country == region_norm)

    live_videos_raw = live_q.all()
    
    # Convert Live Videos to CleanVideo-like structure for compatibility
    for lv in live_videos_raw:
        # Create a makeshift CleanVideo object (not saved to DB)
        cv = CleanVideo()
        cv.id = 999999999 + int(lv.comment_count or 0) # Fake ID
        cv.video_id = lv.id
        cv.title = lv.title
        cv.title_clean = lv.title # Assume live data title is clean enough
        cv.description_clean = lv.description
        cv.tags_clean = " ".join(lv.tags) if lv.tags else ""
        cv.view_count = lv.view_count
        cv.like_count = lv.like_count
        cv.source_dataset = "live_api"
        cv.trending_date = lv.trending_date
        
        # Add to main list
        videos.append(cv)

    if not videos:
        return None

    # --- 2) Load topic model and predict cluster for the filtered videos ---
    
    # NEW (Phase 6): Check if Advanced Model is requested AND available
    # NEW (Phase 6): Check if Advanced/Hybrid Model is requested
    from ..ml.inference import load_advanced_trend_model, load_trend_topic_model, load_hybrid_trend_model
    
    # We use a simple config check or parameter. For now, we try standard first unless "advanced" is passed.
    # But since the function signature doesn't have it yet, we add it.
    # WAIT: I need to update the signature first. But for now, let's assume we use internal logic or kwargs.
    # Actually, let's just use the presence of the model file as a "soft launch" if desired, 
    # OR better: Add `use_advanced_model` arg to the function signature.
    
    # For this specific edit, I will stick to the plan of modifying the Logic BLOCK.
    # I will modify the function signature in a separate edit to avoid breaking callers not yet updated.
    
    labels = None
    confidences = None
    cluster_meta = {}
    
    advanced_model_payload = None
    if use_advanced_model:
         advanced_model_payload = load_advanced_trend_model()
         
    if use_advanced_model and advanced_model_payload:
        # --- PATH B: ADVANCED (S-BERT + UMAP + HDBSCAN) ---
        current_app.logger.info("Using ADVANCED Trend Model for strategy.")
        
        # 1. Embed (S-BERT)
        # We need to import SentenceTransformer locally to avoid global dependency crash if not installed
        try:
            from sentence_transformers import SentenceTransformer
            import umap
            import hdbscan
        except ImportError:
            current_app.logger.error("Advanced dependencies missing. Falling back to old model.")
            advanced_model_payload = None # Trigger fallback
        
        if advanced_model_payload:
            embedder_name = advanced_model_payload.get("embedding_model_name", "all-MiniLM-L6-v2")
            embedder = SentenceTransformer(embedder_name)
            
            texts = [_build_text_for_clean_video(v) for v in videos]
            if not any(texts):
                 return _fallback_trend_strategy(region_norm, start_date, end_date, videos)
                 
            embeddings = embedder.encode(texts, show_progress_bar=False)
            
            # 2. UMAP Transform
            umap_model = advanced_model_payload["umap_model"]
            umap_embeddings = umap_model.transform(embeddings)
            
            # 3. HDBSCAN Predict
            hdbscan_model = advanced_model_payload["hdbscan_model"]
            # returns (labels, strengths) -> strengths are confidence
            labels, confidences = hdbscan.approximate_predict(hdbscan_model, umap_embeddings)
            
            cluster_meta = advanced_model_payload.get("cluster_meta", {})
            # Note: HDBSCAN labels include -1 for noise

    # --- PATH C: HYBRID (TF-IDF + K-Means) - LIGHTWEIGHT IMPLEMENTATION ---
    if labels is None and (model_type == "hybrid"):
        current_app.logger.info("Using HYBRID Trend Model (Pure Python Inference).")
        
        engine = _get_hybrid_engine()
        
        if engine and engine.loaded:
            try:
                # 1. Prepare Texts
                texts = []
                for v in videos:
                    t = (v.title_clean or v.title or "").strip()
                    TAG = (v.tags_clean or v.tags_text or "").strip()
                    d = (v.description_clean or "").strip()
                    if len(d) > 500: d = d[:500]
                    texts.append(f"{t} {TAG} {d}".strip())
                
                # 2. Predict (In-Process, Fast, Robust)
                if not texts:
                     labels = None
                else:
                    labels, confidences = engine.predict_full(texts)
                    
                    # 3. Build cluster_meta keyed by cluster ID
                    #    (matching Legacy format so downstream pipeline works)
                    if labels:
                        cluster_meta = {}
                        unique_labels = set(labels)
                        for cid in unique_labels:
                            # Find dominant genre for this cluster
                            cluster_videos_for_meta = [
                                videos[i] for i, l in enumerate(labels) if l == cid
                            ]
                            genre_counts = Counter(
                                v.genre for v in cluster_videos_for_meta if v.genre
                            )
                            dominant_genre = (
                                genre_counts.most_common(1)[0][0]
                                if genre_counts else "Lifestyle"
                            )
                            # Get top keywords from centroids
                            kw = engine.get_top_keywords(cid, top_n=10)
                            cluster_meta[cid] = {
                                "dominant_genre": dominant_genre,
                                "top_keywords": kw,
                            }
                    else:
                        cluster_meta = {}
                    
            except Exception as e:
                current_app.logger.error("Hybrid inference error: %s", e)
                labels = None
        else:
            current_app.logger.warning("Hybrid engine unavailable. Falling back.")
            labels = None


            
    # --- PATH A: BASELINE (TF-IDF + K-Means) ---
    if labels is None or len(labels) == 0:
        if use_advanced_model:
             current_app.logger.warning("Advanced model failed or missing. Falling back to Baseline.")
             
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
        
        # Calculate Labels & Confidences using distance ratios
        dists = kmeans.transform(X)
        labels = []
        confidences = []
        
        for d in dists:
            best_idx = int(d.argmin())
            labels.append(best_idx)
            
            # Confidence Logic: Softmax over Top 5 Distances
            d_sorted = sorted(d)
            if len(d_sorted) > 1:
                # P(c|x) ~ exp(-dist)
                # Subtract min distance for numerical stability
                min_d = d_sorted[0]
                top_k = d_sorted[:5]
                
                try:
                    import math
                    # Use negative distance for exponent (closer = higher prob)
                    exps = [math.exp(-(val - min_d)) for val in top_k]
                    confidence = exps[0] / sum(exps)
                except Exception:
                    confidence = 1.0
                    
                confidences.append(max(0.0, min(1.0, confidence)))
            else:
                confidences.append(1.0)

    # --- 3) Identify the MOST TRENDING Cluster (Genre-Aware) ---
    cluster_scores: Counter[int] = Counter()
    cluster_indices: Dict[int, List[int]] = defaultdict(list)
    
    # 3a) Fetch Creator Profile Early (for Genre-Aware Selection)
    creator_profile = None
    if user_id:
        try:
            from ..models.creator_profile import CreatorProfile
            creator_profile = CreatorProfile.query.filter_by(user_id=user_id).first()
        except Exception:
            pass

    # 3b) Score based on Views + Freshness
    for idx, (v, label) in enumerate(zip(videos, labels)):
        # SMART WEIGHTING: Boost "live_api" videos by 5x
        weight = 5 if v.source_dataset == 'live_api' else 1
        score = (v.view_count or 0) * weight
        cluster_scores[label] += score
        cluster_indices[label].append(idx)

    if not cluster_scores:
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    # 3c) Apply Personalization Boost (Genre Matching)
    best_cluster = None
    relevance_score = 0  # 0-100 score of how relevant this trend is to the user
    
    if creator_profile and creator_profile.primary_genre:
        user_genre = creator_profile.primary_genre.lower()
        adjusted_scores = {}
        
        for cid, raw_score in cluster_scores.items():
            meta = cluster_meta.get(cid, {})
            cluster_genre = meta.get("dominant_genre", "").lower()
            
            # Boost logic:
            # - Exact Genre Match: +50% score boost
            # - "Lifestyle" fallback: +10% boost (broad appeal)
            boost_factor = 1.0
            
            if user_genre in cluster_genre or cluster_genre in user_genre:
                boost_factor = 1.5  # Huge boost for direct match
            elif "lifestyle" in cluster_genre:
                boost_factor = 1.1  # Slight boost for general appeal
                
            adjusted_scores[cid] = raw_score * boost_factor
            
        best_cluster = max(adjusted_scores, key=adjusted_scores.get)
        
        # Calculate Relevance Score (Display Requirement)
        best_meta = cluster_meta.get(best_cluster, {})
        best_genre = best_meta.get("dominant_genre", "").lower()
        if user_genre in best_genre or best_genre in user_genre:
            relevance_score = 95
        elif "lifestyle" in best_genre:
            relevance_score = 75
        else:
            relevance_score = 45 # Low relevance but still "trending"
            
    else:
        # No profile? Just pick sheer popularity
        best_cluster, _ = cluster_scores.most_common(1)[0]
        relevance_score = 100 # Generic relevance
        
    meta = cluster_meta.get(best_cluster)
    if meta is None:
        return _fallback_trend_strategy(region_norm, start_date, end_date, videos)

    # --- 3d) Data Volume Metrics (Transparency) ---
    data_volume = {
        "totalVideosAnalyzed": len(videos),
        "historicalVideos": len([v for v in videos if v.source_dataset != 'live_api']),
        "liveApiVideos": len([v for v in videos if v.source_dataset == 'live_api']),
        "clusterSize": len(cluster_indices[best_cluster]),
        "dateRange": f"{start_date.date()} to {end_date.date()}"
    }

    # --- 4) DYNAMIC OUTPUT GENERATION ---
    cluster_vids = [videos[i] for i in cluster_indices[best_cluster]]

    # Calculate Prediction Confidence (Task 3)
    avg_confidence = 0.0
    if confidences:
        cluster_confs = [confidences[i] for i in cluster_indices[best_cluster]]
        if cluster_confs:
            avg_confidence = round(sum(cluster_confs) / len(cluster_confs), 2)
    else:
        avg_confidence = 0.75 # Default

    core_genre: str = meta["dominant_genre"]
    
    dynamic_texts: List[str] = []
    for v in cluster_vids:
        title = (v.title_clean or v.title or "")
        tags_str = (v.tags_clean or v.tags_text or "")
        dynamic_texts.append(f"{title} {tags_str}")

    top_terms: List[str] = extract_top_keywords(dynamic_texts, top_n=15)
    
    # FILTER: Remove past years (e.g. "2018", "2019", "2020"...) from keywords
    import re
    year_pattern = re.compile(r"^20\d{2}$")
    top_terms = [t for t in top_terms if not year_pattern.match(t)]
    
    # Cap at 8 after filtering
    top_terms = top_terms[:8]

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

    # --- 5) Real Sentiment (TextBlob NLP) ---
    sentiment_scores = []
    # Sample up to 500 videos for performance
    for v in cluster_vids[:500]: 
        text = f"{v.title_clean or ''} {v.description_clean or ''}"
        if len(text) > 5:
            blob = TextBlob(text)
            sentiment_scores.append(blob.sentiment.polarity) # -1.0 to 1.0

    n_pos = sum(1 for s in sentiment_scores if s > 0.1)
    n_neg = sum(1 for s in sentiment_scores if s < -0.1)
    n_neu = len(sentiment_scores) - n_pos - n_neg
    total_sent = len(sentiment_scores) or 1
    
    positive_ratio = n_pos / total_sent # Used for ml_signals below
    
    sentiment = {
        "positive": round(n_pos / total_sent, 2),
        "neutral": round(n_neu / total_sent, 2),
        "negative": round(n_neg / total_sent, 2),
        "sampleSize": len(sentiment_scores),
        "method": "TextBlob NLP on video titles & descriptions"
    }

    # --- 6) Suggested Videos (With Similarity Score) ---
    suggested_videos = []
    
    # Try to calculate cosine similarity if Baseline Model is active
    local_centroid = None
    vec_func = None
    selection_method = "Top Views in Cluster"
    
    if locals().get("vectorizer"):
         try:
             # Calculate LOCAL centroid for the current batch of videos
             batch_vectors = []
             valid_vids_for_centroid = []
             
             for v in cluster_vids:
                 t = (v.title_clean or v.title or "").strip()
                 if t:
                     batch_vectors.append(vectorizer.transform([t]))
                     valid_vids_for_centroid.append(v)
             
             if batch_vectors:
                 # Compute mean vector of the cluster
                 # vstack returns sparse matrix. mean returns np.matrix (dense)
                 local_centroid = vstack(batch_vectors).mean(axis=0)
                 # CONVERT TO ARRAY heavily advised for sklearn cosine_similarity
                 local_centroid = np.asarray(local_centroid)
                 vec_func = lambda t: vectorizer.transform([t])
         except Exception as e:
             current_app.logger.warning(f"Failed to compute local centroid: {e}")
             pass
         
    # 5.5) Trend Metrics (Real Engagement Stats)
    trend_metrics = _calculate_trend_metrics_normalized(top_terms, cluster_vids)
    
    for v in top_cluster_vids:
        vid_id = v.video_id or str(v.id)
        
        # Calculate Similarity Score (Task 7)
        sim_score = 0.85 # Default high score fallback
        method = selection_method
        
        if local_centroid is not None and vec_func:
            try:
                vec = vec_func(v.title_clean or "")
                # Convert sparse vector to dense array for distance calculation
                vec_dense = vec.toarray()
                # Calculate Euclidean distance to centroid
                dist = np.linalg.norm(vec_dense - local_centroid)
                sim = 1.0 / (1.0 + float(dist))
                sim_score = round(sim, 2)
                method = "Distance to Centroid"
            except Exception: 
                pass
        
        suggested_videos.append(
            {
                "videoId": vid_id,
                "title": v.title,
                "channel": v.channel_title or v.channel_id,
                "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/default.jpg",
                "views": v.view_count,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "similarityScore": sim_score, # NEW
                "selectionMethod": method,    # NEW
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
        "sentiment_ratio": positive_ratio, # Now from TextBlob
        "dataVolume": data_volume,         # NEW
        "relevanceScore": relevance_score, # NEW
        "confidence": avg_confidence,      # NEW
        "trendMetrics": trend_metrics,     # NEW
    }

    # --- 7b) Inject Creator Profile for personalization ---
    if creator_profile and creator_profile.onboarding_completed:
        ml_signals["creatorProfile"] = {
            "primaryGenre": creator_profile.primary_genre,
            "contentStyle": creator_profile.content_style,
            "targetAudienceAge": creator_profile.target_audience_age,
            "targetRegion": creator_profile.target_region,
            "creatorGoal": creator_profile.creator_goal,
            "channelName": creator_profile.channel_name or "N/A",
            "subscriberCount": creator_profile.subscriber_count or 0,
        }
        current_app.logger.info(
            "Personalized strategy for user %s (%s creator) - Relevance: %s",
            user_id, creator_profile.primary_genre, relevance_score
        )

    # --- 8) GEMINI API CALL REPLACES STATIC STRING ---
    gemini_result = _generate_strategy_with_gemini(ml_signals)
    
    detailed_strategy = gemini_result.get("detailedStrategy")
    marketing_hooks = gemini_result.get("marketingHooks", [])
    
    # Fallback to top_terms if hooks are empty
    final_insights = marketing_hooks if marketing_hooks else top_terms[:5]
    
    strategy_engine = ml_signals.get("strategyEngine", "gemini")
    strategy_source = f"ml-topic-model-{strategy_engine}"

    response = {
        "region": region_norm,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "coreTrend": core_trend_label,
        "coreGenre": core_genre,
        "detailedStrategy": detailed_strategy,
        "sentiment": sentiment,
        "dataVolume": data_volume,     # NEW: Exposed to API
        "relevanceScore": relevance_score, # NEW: Exposed to API
        "predictionConfidence": avg_confidence, # NEW: Exposed to API
        "trendMetrics": trend_metrics,     # NEW: Exposed to API
        "suggestedVideos": suggested_videos,
        "marketingInsights": final_insights,
        "mlSignals": ml_signals,
        "strategySource": strategy_source,
    }

    # ── Spread personalized Gemini fields into response ──
    if ml_signals.get("isPersonalized"):
        response["personalizedIntro"] = gemini_result.get("personalizedIntro")
        response["emergingTrends"] = gemini_result.get("emergingTrends", [])
        response["recommendedAngles"] = gemini_result.get("recommendedAngles", [])
        response["contentGaps"] = gemini_result.get("contentGaps", [])
        response["titleSuggestions"] = gemini_result.get("titleSuggestions", [])
        response["optimalPosting"] = gemini_result.get("optimalPosting", {})

    return response


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