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

# GROQ INTEGRATION START
from groq import Groq

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Global variables for the model and client instance (start as None)
_GROQ_CLIENT_INSTANCE = None
GROQ_MODEL = "llama-3.3-70b-versatile"

def _get_groq_client():
    """
    Lazily initializes the Groq client using GROQ_API_KEY from environment.
    """
    global _GROQ_CLIENT_INSTANCE

    if _GROQ_CLIENT_INSTANCE is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            current_app.logger.error("FATAL: GROQ_API_KEY environment variable is missing or empty.")
            return None

        try:
            _GROQ_CLIENT_INSTANCE = Groq(api_key=api_key)
        except Exception as e:
            current_app.logger.error("Groq Client initialization failed: %s", str(e))
            _GROQ_CLIENT_INSTANCE = None

    return _GROQ_CLIENT_INSTANCE

# GROQ INTEGRATION END


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

        # Optional "followed by …" for the next 1–2 genres
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

def _build_strategy_from_signals(ml_signals: Dict) -> list:
    """
    Fallback: build 3 punchy creator-friendly strategy bullet points using only ML signals,
    shown when Groq is unavailable or fails.
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
            keywords_str = f"{primary_kw}, {other_kws}"
        else:
            keywords_str = primary_kw
    else:
        keywords_str = "the most searched terms in your niche"

    # Period text
    if months:
        month_labels = ", ".join(
            month_name[m] for m in months if isinstance(m, int) and 1 <= m <= 12
        )
        period_str = f"in {month_labels}"
    else:
        period_str = "over the upcoming weeks"

    # Sentiment description
    sentiment_ratio = float(ml_signals.get("sentiment_ratio", 0.0) or 0.0)
    if sentiment_ratio >= 0.9:
        sentiment_desc = "Audience is highly receptive — consistent uploads will compound fast"
    elif sentiment_ratio >= 0.8:
        sentiment_desc = "Strong viewer interest detected — lean into a short content series"
    elif sentiment_ratio >= 0.7:
        sentiment_desc = "Solid growing interest — strong hooks and clear value will stand out"
    else:
        sentiment_desc = "Viewers are selective right now — prioritise clarity and a strong opening"

    return [
        f"Target {region} creators are gravitating toward {core_genre} content {period_str} — lead with {primary_kw if top_keywords else 'your niche keyword'} in your title.",
        f"Build videos around search-friendly topics: {keywords_str}. Tutorials, comparisons, and quick explainers perform best for this trend.",
        sentiment_desc + ".",
    ]


# GROQ INTEGRATION START
def _generate_strategy_with_gemini(ml_signals: Dict) -> Dict:
    """
    Uses the Groq API (Llama 3.3 70B) to turn structured ML signals into a STRUCTURED,
    personalized strategy with actionable bullet points.
    """
    client = _get_groq_client()
    if client is None:
        current_app.logger.warning("Groq client is None — check GROQ_API_KEY env var.")
        _is_pers = bool(ml_signals.get("creatorProfile"))
        ml_signals["strategyEngine"] = "ml-fallback"
        ml_signals["isPersonalized"] = _is_pers
        _core_trend = ml_signals.get("coreTrend") or ml_signals.get("coreGenre") or "this niche"
        _fallback = {
            "strategyPoints": _build_strategy_from_signals(ml_signals),
            "coreTrendExplanation": f"Creators in the {_core_trend} space are seeing measurable growth in this region and period.",
            "marketingHooks": [],
            "videoIdeas": [],
            "descriptionKeywords": [],
        }
        if _is_pers:
            _fallback.update({"personalizedIntro": None, "emergingTrends": [], "contentGaps": [], "titleSuggestions": [], "contentTone": {}, "channelSpecificTips": [], "optimalPosting": {}})
        return _fallback

    positive_ratio = ml_signals.get("sentiment_ratio", 0.0)
    sentiment_level = "highly positive" if positive_ratio > 0.95 else "strong positive"

    # Confidence-aware framing
    from ..ml.niche_filter import get_confidence_tier
    _raw_confidence = ml_signals.get("confidence", 0.5)
    _conf_tier = get_confidence_tier(_raw_confidence)
    _conf_label = _conf_tier["label"]
    _conf_tier_name = _conf_tier["tier"].upper()

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    current_year = today.year

    # ── Format analysis period for prompts ──
    from datetime import datetime as _dt
    _raw_start = ml_signals.get("analysisStartDate")
    _raw_end = ml_signals.get("analysisEndDate")
    try:
        analysis_start = _dt.fromisoformat(_raw_start).strftime("%B %Y") if _raw_start else today_str
        analysis_end = _dt.fromisoformat(_raw_end).strftime("%B %Y") if _raw_end else today_str
    except (ValueError, TypeError):
        analysis_start = today_str
        analysis_end = today_str

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

CONFIDENCE LEVEL: {conf_tier_name} ({conf_label}) — Raw score: {raw_confidence:.0%}

CRITICAL RULES:
1. NEVER mention the creator's name or channel name. Address them as "you/your".
2. This is THEIR dashboard — speak directly, casually, like a smart advisor friend.
3. ALL recommendations MUST be relevant to the creator's declared genre ({genre}). Never suggest off-niche topics.
4. Cross-reference trending keywords with THEIR genre creatively.
   Example: If "Food" is trending and they do Travel, suggest "Street Food Tours", "Culinary Destination Guides".
5. Be specific with data signals (use +XX%, "rising keyword", "crossing over from X").
6. Content gaps should feel like insider advice: "You haven't covered X, but it's the #N rising keyword..."
7. Title suggestions must show a BAD generic title vs YOUR optimized version with trending keywords.
8. CONFIDENCE RULES — apply based on the confidence level above:
   - HIGH (>=70%): Use confident, direct recommendations ("This will perform well", "Strong signal for...").
   - MEDIUM (40-69%): Use moderate language ("likely to trend", "showing growth", "consider testing...").
   - LOW (<40%): Frame ALL suggestions as exploratory ("early signal suggests", "worth testing", "experimental idea"). Add a brief note that data signals are weak for this period.

OUTPUT FORMAT (JSON ONLY):
{{
  "coreTrendExplanation": "One punchy sentence, MAX 15 words, NO numbers or percentages. Describe the creator behaviour shift only. e.g. 'Creators blending challenge content with daily life moments are gaining rapid traction.'",
  "personalizedIntro": "1-2 sentence punchy intro with emoji. MUST reference the analysis period ({analysis_start} to {analysis_end}), NOT today's date. MUST mention the region ({region}) and genre ({genre}). Example: '🎯 Here\\'s your personalized strategy for {analysis_start} to {analysis_end} — 3 emerging opportunities in the {region} {genre} space match your niche perfectly.'",
  "emergingTrends": [
    {{
      "trend": "Name of the trend",
      "signal": "Short data signal, e.g. '+45% search volume' or 'crossing over from Lifestyle into Travel'",
      "relevanceToYou": "Why this matters for YOUR genre. 1 sentence, speak directly."
    }}
  ],
  "videoIdeas": [
    {{
      "title": "A specific, filmable video title for this creator",
      "hook": "The first 15-second hook — what should the creator say/show to instantly hook viewers",
      "trendKeyword": "Which trending keyword this idea is built around"
    }}
  ],
  "titleSuggestions": [
    {{
      "draft": "A generic title a creator might use (bad example)",
      "optimized": "The SEO-optimized version with trending keywords baked in",
      "whyBetter": "Which trending keywords were added and why they boost CTR"
    }}
  ],
  "descriptionKeywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"],
  "contentTone": {{
    "recommended": "e.g. 'Energetic & motivational' or 'Educational & calm' or 'Humorous & relatable'",
    "reason": "1 sentence: why this tone works for their genre + audience age"
  }},
  "contentGaps": [
    {{
      "topic": "A trending topic you should cover but likely haven't",
      "insight": "Data-backed reason, e.g. 'It\\'s the #2 rising keyword in your niche. Creators covering this got 2.5x more views.'"
    }}
  ],
  "channelSpecificTips": [
    {{
      "tip": "A specific actionable advice item tailored to their subscriber level ({subs} subs) and goal ({goal})",
      "action": "The exact next step — what they should do TODAY"
    }}
  ],
  "optimalPosting": {{
    "bestDays": "e.g. 'Tuesdays & Thursdays'",
    "bestTime": "e.g. '2pm - 5pm EST'",
    "reason": "Brief reason based on audience age + region demographics"
  }},
  "marketingHooks": ["Hook 1", "Hook 2", "Hook 3", "Hook 4", "Hook 5"],
  "strategyPoints": [
    "Tip 1: 15-25 words, punchy and actionable — content format angle",
    "Tip 2: 15-25 words, punchy and actionable — SEO/keyword tactic",
    "Tip 3: 15-25 words, punchy and actionable — posting or engagement tip"
  ],
  "detailedStrategy": "A concise 2-3 sentence summary tying everything together. Speak directly to the creator."
}}

RULES:
- JSON only. No markdown. No code fences.
- emergingTrends: exactly 3 items.
- videoIdeas: exactly 3 items. Each idea must be directly filmable by this creator today.
- titleSuggestions: exactly 2 examples.
- descriptionKeywords: exactly 8 SEO keywords (no hashtags, no #, just words/phrases).
- contentTone: exactly 1 object with recommended + reason.
- contentGaps: exactly 2 items. Reference trending keywords related to their genre.
- channelSpecificTips: exactly 2 items tailored to their subscriber level and goal.
- marketingHooks: exactly 5 punchy multi-word phrases.
- strategyPoints: exactly 3 items, each 15-25 words, punchy and direct. Tailored to this creator's genre and goal.
- optimalPosting: Base on their audience age ({audience_age}) and region ({audience_region}).
- CRITICAL: NEVER use placeholders of any kind. This includes: bracket placeholders [Game], [Topic], [e.g. ...] AND bare letter placeholders like "X", "Y", "Z" used as stand-ins for real words. Always write the actual specific word. Bad: "The Ultimate Guide to Mastering X" or "My [Game] Journey". Good: "The Ultimate Guide to Mastering Elden Ring" or "My Minecraft Journey".
"""
        prompt_vars = dict(
            signals=json.dumps(
                {k: v for k, v in ml_signals.items() if k != "creatorProfile"},
                indent=2,
                cls=_NumpyEncoder,
            ),
            region=ml_signals.get("effectiveRegion", "Unknown"),
            months=ml_signals.get("effectiveMonths", "Unknown"),
            today_str=today_str,
            current_year=current_year,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            genre=genre,
            style=style,
            subs=subs,
            goal=goal,
            audience_age=audience_age,
            audience_region=audience_region,
            conf_tier_name=_conf_tier_name,
            conf_label=_conf_label,
            raw_confidence=_raw_confidence,
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
Predict the MODERN trend for {current_year} and give actionable content ideas any creator can use.

OUTPUT FORMAT (JSON ONLY):
{{
  "coreTrendExplanation": "One punchy sentence, MAX 15 words, NO numbers or percentages. Describe the creator behaviour shift only. e.g. 'Creators blending challenge content with daily life moments are gaining rapid traction.'",
  "strategyPoints": [
    "Tip 1: 15-25 words, punchy and actionable",
    "Tip 2: 15-25 words, punchy and actionable",
    "Tip 3: 15-25 words, punchy and actionable"
  ],
  "videoIdeas": [
    {{
      "title": "A specific, filmable video title",
      "hook": "The first 15-second hook to instantly capture attention",
      "trendKeyword": "Which trending keyword this idea is built around"
    }}
  ],
  "descriptionKeywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"],
  "marketingHooks": ["Hook 1", "Hook 2", "Hook 3", "Hook 4", "Hook 5"]
}}

RULES:
- JSON only. No markdown. No code fences.
- coreTrendExplanation: exactly 1 sentence, 15-25 words. Explain what the trend means for creators, not what the label says.
- strategyPoints: exactly 3 items. Each 15-25 words, direct and actionable. Cover: (1) content format angle, (2) SEO/keyword tactic, (3) posting/engagement tip.
- videoIdeas: exactly 3 items. Each should be concrete and filmable today.
- descriptionKeywords: exactly 8 SEO keywords or short phrases (no # symbols).
- marketingHooks: exactly 5 punchy multi-word phrases (e.g., "Cozy Gaming Setup", "ASMR Study Hour").
- CRITICAL: NEVER use placeholders of any kind. This includes: bracket placeholders [Topic], [Game], [e.g. ...] AND bare letter stand-ins like "X", "Y", "Z". Always write the actual specific word. Bad: "Trying [Skill] for the First Time" or "Guide to Mastering X". Good: "Trying Pottery for the First Time" or "Guide to Mastering Sourdough Bread".
"""
        prompt_vars = dict(
            signals=json.dumps(
                {k: v for k, v in ml_signals.items() if k != "creatorProfile"},
                indent=2,
                cls=_NumpyEncoder,
            ),
            region=ml_signals.get("effectiveRegion", "Unknown"),
            months=ml_signals.get("effectiveMonths", "Unknown"),
            sentiment_level=sentiment_level,
            today_str=today_str,
            current_year=current_year,
        )

    try:
        prompt = prompt_template.format(**prompt_vars)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)

        ml_signals["strategyEngine"] = "groq"
        ml_signals["isPersonalized"] = is_personalized
        return result

    except Exception as e:
        current_app.logger.error("Groq API call failed: %s", e, exc_info=True)
        ml_signals["strategyEngine"] = "ml-fallback"
        ml_signals["isPersonalized"] = is_personalized
        _core_trend = ml_signals.get("coreTrend") or ml_signals.get("coreGenre") or "this niche"
        _fallback = {
            "strategyPoints": _build_strategy_from_signals(ml_signals),
            "coreTrendExplanation": f"Creators in the {_core_trend} space are seeing measurable growth in this region and period.",
            "marketingHooks": [],
            "videoIdeas": [],
            "descriptionKeywords": [],
        }
        if is_personalized:
            _fallback.update({"personalizedIntro": None, "emergingTrends": [], "contentGaps": [], "titleSuggestions": [], "contentTone": {}, "channelSpecificTips": [], "optimalPosting": {}})
        return _fallback

# GROQ INTEGRATION END

from scipy.sparse import vstack # NEW: For local centroid calculation
from sqlalchemy import func, extract
from flask import current_app

# ... imports ...


# ---------------------------------------------------------------------------
# 2. TREND STRATEGY (Trend Strategy section)
# ---------------------------------------------------------------------------

def _calculate_trend_metrics_normalized(
    keywords: List[str],
    cluster_vids: List['CleanVideo'],
    all_videos: List['CleanVideo'] = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> Dict:
    """
    Computes real engagement metrics for the top keywords in the cluster using 0-100 normalized scores.
    Returns: { keyword: { volume, avgViews, frequency, growth, trendScore } }

    Growth is calculated by splitting all_videos (full date-range pool) at the date
    midpoint into an OLD and NEW window, then measuring how keyword frequency changed.
    Using the full pool (vs. just cluster_vids) gives far more data per window and
    eliminates the 0.0 growth bug caused by sparse cluster sizes.
    """
    metrics = {}
    n_vids = len(cluster_vids)
    if n_vids == 0:
        return {}

    # --- Build time-window split for growth calculation ---
    # Prefer the full video pool (all_videos) over just the cluster for stability.
    # IMPORTANT: Always split on the ACTUAL data's date range, never on the requested
    # future date range. Historical data is from 2018-2020; if we split at 2026-05-01
    # every video lands in "old", new_vids is empty, and growth = -100.0 for all.
    growth_pool = all_videos if (all_videos and len(all_videos) >= 20) else cluster_vids

    actual_dates = [v.trending_date for v in growth_pool if v.trending_date is not None]
    if len(actual_dates) >= 2:
        actual_min = min(actual_dates)
        actual_max = max(actual_dates)
        midpoint = actual_min + (actual_max - actual_min) / 2
        old_vids = [v for v in growth_pool if v.trending_date and v.trending_date < midpoint]
        new_vids = [v for v in growth_pool if v.trending_date and v.trending_date >= midpoint]
    else:
        dated_vids = sorted(
            growth_pool,
            key=lambda v: v.trending_date if v.trending_date is not None else datetime.min
        )
        cutoff = max(1, len(dated_vids) // 2)
        old_vids = dated_vids[:cutoff]
        new_vids = dated_vids[cutoff:]

    n_old = len(old_vids)
    n_new = len(new_vids)

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

        # --- Growth calculation: compare keyword frequency across time windows ---
        old_matches = sum(
            1 for v in old_vids
            if kw_lower in (v.title_clean or "").lower()
            or kw_lower in (v.description_clean or "").lower()
        )
        new_matches = sum(
            1 for v in new_vids
            if kw_lower in (v.title_clean or "").lower()
            or kw_lower in (v.description_clean or "").lower()
        )
        freq_old = old_matches / n_old if n_old > 0 else 0.0
        freq_new = new_matches / n_new if n_new > 0 else 0.0

        if freq_old > 0:
            # Percentage change from old window to new window
            raw_growth = ((freq_new - freq_old) / freq_old) * 100.0
        elif freq_new > 0:
            # Keyword only appeared in newer window — treat as 100% growth
            raw_growth = 100.0
        else:
            raw_growth = 0.0

        # Cap growth to a reasonable display range
        growth = round(max(-100.0, min(200.0, raw_growth)), 1)

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
            "frequency": freq,
            "growth": growth,
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

    # ── Early Creator Profile Fetch (needed for niche pre-filtering) ──
    creator_profile = None
    if user_id:
        try:
            from ..models.creator_profile import CreatorProfile
            creator_profile = CreatorProfile.query.filter_by(user_id=user_id).first()
        except Exception:
            pass

    # --- 1) Base query: Filter historical data by season (month) and region ---
    # Exclude most_popular dataset: it contains creator-name-heavy content that
    # was not included during model training (retrain_with_svd.py excludes it too).
    # Including it at inference contaminates keyword extraction with niche creator
    # names (e.g. "sturniolo", "jmancurly") instead of meaningful trend signals.
    base_q = db.session.query(CleanVideo).filter(
        CleanVideo.trending_date.isnot(None),
        extract("month", CleanVideo.trending_date).in_(month_list),
        CleanVideo.source_dataset != 'most_popular',
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

    # ── Niche Pre-Filter: Remove off-niche videos before clustering ──
    # This prevents music videos, cat videos, etc. from polluting trend signals
    # for niche creators (travel, gaming, etc.)
    if creator_profile and creator_profile.primary_genre:
        from ..ml.niche_filter import filter_videos_by_niche
        pre_filter_count = len(videos)
        videos = filter_videos_by_niche(
            videos,
            creator_profile.primary_genre,
            # threshold default is now 0.33 (Phase 3 fix — was 0.1)
            # min_keep fallback now never includes score=0.0 videos
        )
        current_app.logger.info(
            "Niche pre-filter (%s): %d → %d videos",
            creator_profile.primary_genre, pre_filter_count, len(videos),
        )

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


            
    # --- PATH SVD: TF-IDF → SVD (200 dims) → K-Means (Phase 2B fix) ---
    # Used when model_type == "svd" OR when model_type == "legacy" and SVD model exists.
    # This replaces the broken 30K-dim K-Means with a proper dimensionality-reduced version.
    if labels is None and model_type in ("svd", "legacy"):
        from ..ml.inference import load_svd_trend_model
        svd_payload = load_svd_trend_model()
        if svd_payload:
            current_app.logger.info("Using SVD trend model (200-dim LSA + K-Means).")
            vectorizer = svd_payload["vectorizer"]
            svd        = svd_payload["svd"]
            normalizer = svd_payload["normalizer"]
            kmeans_svd = svd_payload["kmeans"]
            cluster_meta = svd_payload["cluster_meta"]

            texts = [_build_text_for_clean_video(v) for v in videos]
            if any(texts):
                X_tfidf = vectorizer.transform(texts)
                X_svd   = svd.transform(X_tfidf)
                X_norm  = normalizer.transform(X_svd)
                dists   = kmeans_svd.transform(X_norm)

                labels = []
                confidences = []
                for d in dists:
                    best_idx = int(d.argmin())
                    labels.append(best_idx)
                    d_sorted = sorted(d)
                    if len(d_sorted) > 1:
                        try:
                            import math
                            min_d = d_sorted[0]
                            top_k = d_sorted[:5]
                            exps = [math.exp(-(val - min_d)) for val in top_k]
                            confidence = exps[0] / sum(exps)
                        except Exception:
                            confidence = 1.0
                        confidences.append(max(0.0, min(1.0, confidence)))
                    else:
                        confidences.append(1.0)

    # --- PATH LDA: CountVectorizer → LDA (Phase 2C comparison model) ---
    # Used when model_type == "lda".
    if labels is None and model_type == "lda":
        from ..ml.inference import load_lda_model
        lda_payload = load_lda_model()
        if lda_payload:
            current_app.logger.info("Using LDA topic model.")
            lda_vectorizer = lda_payload["vectorizer"]
            lda_model      = lda_payload["lda"]
            cluster_meta   = lda_payload["cluster_meta"]

            texts = [_build_text_for_clean_video(v) for v in videos]
            if any(texts):
                X_counts  = lda_vectorizer.transform(texts)
                X_topics  = lda_model.transform(X_counts)   # (n_docs, n_topics)
                labels      = X_topics.argmax(axis=1).tolist()
                confidences = X_topics.max(axis=1).tolist()

    # --- PATH A: BASELINE (TF-IDF + K-Means, prefers Elbow-optimised model) ---
    if labels is None or len(labels) == 0:
        if use_advanced_model:
            current_app.logger.warning("Advanced model failed or missing. Falling back to Baseline.")

        # Prefer the Elbow-validated optimised model if available
        from ..ml.inference import load_optimised_trend_model
        topic_payload = load_optimised_trend_model() or load_trend_topic_model()
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
    
    # 3a) Creator Profile already fetched early (above) for niche pre-filtering
    # creator_profile is already set; no second DB query needed

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

    # Enforce genre lock: use creator's declared genre, not cluster's dominant genre.
    # This prevents coreGenre from drifting to "Lifestyle" for Travel/Gaming creators.
    if creator_profile and creator_profile.primary_genre:
        core_genre: str = creator_profile.primary_genre
    else:
        core_genre: str = meta["dominant_genre"]

    dynamic_texts: List[str] = []
    for v in cluster_vids:
        title = (v.title_clean or v.title or "")
        tags_str = (v.tags_clean or v.tags_text or "")
        dynamic_texts.append(f"{title} {tags_str}")

    # Personalized mode gets more initial keywords to survive niche filtering
    _kw_top_n = 20 if (creator_profile and creator_profile.primary_genre) else 15
    _kw_cap   = 12 if (creator_profile and creator_profile.primary_genre) else 8

    top_terms: List[str] = extract_top_keywords(dynamic_texts, top_n=_kw_top_n)

    # FILTER: Remove past years (e.g. "2018", "2019", "2020"...) from keywords
    import re
    year_pattern = re.compile(r"^20\d{2}$")
    top_terms = [t for t in top_terms if not year_pattern.match(t)]

    # Cap before niche filtering
    top_terms = top_terms[:_kw_cap]

    # ── Niche Keyword Filter: Remove off-niche keywords ──
    if creator_profile and creator_profile.primary_genre:
        from ..ml.niche_filter import filter_keywords_by_niche
        _before_niche_filter = top_terms[:]
        top_terms = filter_keywords_by_niche(
            top_terms,
            creator_profile.primary_genre,
            threshold=0.2,   # Lowered from 0.3 — catches partial matches (score=0.7)
            preserve_count=8, # Raised from 5 — keeps more keywords after filtering
        )
        # If niche filter is too aggressive (cluster has diverse/mixed content),
        # fall back to unfiltered keywords — generic gaming terms like "people" or
        # "survive" still carry signal even if not in the Gaming vocabulary.
        if len(top_terms) < 3:
            top_terms = _before_niche_filter

    if not top_terms:
        top_terms = ["trending", core_genre.lower(), "viral"]

    # ── Niche-Aware Video Selection ──
    # Step 1: Hard genre filter using DB_GENRE_MAP (eliminates ASMR/beauty in Gaming)
    # Step 2: Rank survivors by niche score (60%) + normalized views (40%)
    # Step 3: If cluster has no genre-matched videos, query DB directly for that genre
    _from_db_query: set = set()  # tracks video ids that came from direct DB query
    if creator_profile and creator_profile.primary_genre:
        from ..ml.niche_filter import compute_video_niche_score, DB_GENRE_MAP
        _niche_genre = creator_profile.primary_genre

        # Hard filter: only keep videos whose DB genre matches the creator's genre
        _allowed_db_genres = DB_GENRE_MAP.get(_niche_genre, [])
        _candidate_vids = cluster_vids  # default

        if _allowed_db_genres:
            _genre_filtered = [
                v for v in cluster_vids
                if (v.genre or "") in _allowed_db_genres
            ]

            if len(_genre_filtered) >= 3:
                # Cluster has enough genre-matched videos — use them
                _candidate_vids = _genre_filtered
            else:
                # Cluster has no genre-matched videos (K-Means grouped by keyword,
                # not by genre). Query DB directly for this genre.
                try:
                    _db_genre_vids = (
                        db.session.query(CleanVideo)
                        .filter(
                            CleanVideo.genre.in_(_allowed_db_genres),
                            CleanVideo.view_count.isnot(None),
                            CleanVideo.source_dataset != 'most_popular',
                        )
                        .order_by(CleanVideo.view_count.desc())
                        .limit(50)
                        .all()
                    )
                    if _db_genre_vids:
                        _candidate_vids = _db_genre_vids
                        # Mark all DB-queried videos — centroid distance is meaningless for them
                        _from_db_query = {id(v) for v in _db_genre_vids}
                except Exception as _e:
                    current_app.logger.warning("Direct genre DB query failed: %s", _e)

        def _video_rank_score(v):
            niche_s = compute_video_niche_score(v, _niche_genre)
            view_s = min(1.0, (v.view_count or 0) / 5_000_000)
            return niche_s * 0.6 + view_s * 0.4

        top_cluster_vids = sorted(_candidate_vids, key=_video_rank_score, reverse=True)[:5]
    else:
        top_cluster_vids = sorted(
            cluster_vids, key=lambda v: v.view_count or 0, reverse=True,
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
    trend_metrics = _calculate_trend_metrics_normalized(
        top_terms, cluster_vids,
        all_videos=videos,
        start_date=start_date,
        end_date=end_date,
    )

    # ── Filter 0-volume keywords ──
    # extract_top_keywords scans title+tags, but trend_metrics counts only
    # title+description matches. A keyword found only in tags gets volume=0.
    # Only apply this filter if at least 5 keywords survive — for small clusters
    # (e.g. Gaming with 58 videos), most keywords have volume 1-2 and a strict
    # 0-volume cut would reduce topKeywords to 3 or fewer.
    valid_terms = [kw for kw in top_terms if trend_metrics.get(kw, {}).get("volume", 0) > 0]
    if len(valid_terms) >= 5:
        top_terms = valid_terms
        trend_metrics = {kw: trend_metrics[kw] for kw in top_terms}
        core_trend_label = f"{core_genre} — {top_terms[0].title()}"
    # If fewer than 5 survive, keep original list — low-volume keywords are
    # still meaningful signals, just from a small data pool.

    # ── Phase 6: Multi-Signal Confidence Formula ──
    # Replaces the old 2-signal blend (always produced ~0.36).
    # Five independent signals give a meaningful 0.20–0.90 range.
    _cluster_conf = avg_confidence  # save cluster distance signal before overwriting
    _clf_signal = 0.5               # neutral default if classifier unavailable

    try:
        from ..ml.trend_classifier import TrendClassifier
        from ..ml.niche_filter import compute_video_niche_score, get_niche_vocabulary
        _classifier = TrendClassifier()
        if _classifier.load():
            _niche_genre_for_clf = (
                creator_profile.primary_genre if creator_profile else core_genre
            )
            _niche_vocab = get_niche_vocabulary(_niche_genre_for_clf)
            _clf_probs = []
            for _v in cluster_vids[:100]:  # Sample 100 for performance
                _ns = compute_video_niche_score(_v, _niche_genre_for_clf)
                _title = (getattr(_v, 'title_clean', '') or getattr(_v, 'title', '') or '').lower()
                _kd = sum(1 for kw in _niche_vocab if kw in _title) / max(len(_niche_vocab), 1)
                _prob = _classifier.predict_trend_probability({
                    'view_count': _v.view_count or 0,
                    'like_count': _v.like_count or 0,
                    'comment_count': _v.comment_count or 0,
                    'niche_relevance_score': _ns,
                    'cluster_confidence': _cluster_conf,
                    'keyword_density': _kd,
                })
                _clf_probs.append(_prob)
            if _clf_probs:
                _clf_signal = sum(_clf_probs) / len(_clf_probs)
                current_app.logger.info("TrendClassifier signal: %.2f", _clf_signal)
    except Exception as _clf_err:
        current_app.logger.debug("TrendClassifier unavailable: %s", _clf_err)

    # Signal 3: Data volume — cluster size normalised (500 videos = full confidence)
    _volume_signal = min(1.0, data_volume["clusterSize"] / 500.0)

    # Signal 4: Growth — average keyword growth across trend_metrics, mapped [−100,+200] → [0,1]
    # 0% growth maps to 0.5 (neutral), +200% maps to 1.0, −100% maps to 0.0
    if trend_metrics:
        _growth_vals = [m["growth"] for m in trend_metrics.values() if isinstance(m, dict)]
        _avg_growth = sum(_growth_vals) / len(_growth_vals) if _growth_vals else 0.0
        _growth_signal = min(1.0, max(0.0, (_avg_growth + 100.0) / 200.0))
    else:
        _growth_signal = 0.5  # neutral when no metrics

    # Signal 5: Niche relevance (0–100 → 0–1)
    _relevance_signal = (relevance_score or 0) / 100.0

    # Weighted combination → meaningful range (~0.20 low confidence, ~0.88 high confidence)
    avg_confidence = round(
        0.35 * _clf_signal
        + 0.20 * _cluster_conf
        + 0.20 * _volume_signal
        + 0.15 * _growth_signal
        + 0.10 * _relevance_signal,
        2,
    )

    _confidence_breakdown = {
        "classifierProbability": round(_clf_signal, 3),
        "clusterCohesion": round(_cluster_conf, 3),
        "dataVolume": round(_volume_signal, 3),
        "keywordGrowth": round(_growth_signal, 3),
        "nicheRelevance": round(_relevance_signal, 3),
        "weights": {"classifier": 0.35, "cluster": 0.20, "volume": 0.20, "growth": 0.15, "relevance": 0.10},
    }
    current_app.logger.info(
        "Multi-signal confidence: clf=%.2f cluster=%.2f vol=%.2f growth=%.2f rel=%.2f -> %.2f",
        _clf_signal, _cluster_conf, _volume_signal, _growth_signal, _relevance_signal, avg_confidence,
    )

    # --- Pre-compute distances then min-max normalize to [0.50, 0.98] ---
    # 1.0/(1.0+dist) collapses all distances to ~0.5 when K-Means cluster quality
    # is low. Min-max normalization gives meaningful differentiation: the most
    # similar video scores 0.98, the least similar scores 0.50.
    # Compute centroid distances — skip DB-queried videos (they're outside the cluster,
    # their distance is meaningless and always equidistant ≈ 0.5)
    _raw_dists = {}
    if local_centroid is not None and vec_func:
        for _v in top_cluster_vids:
            if id(_v) in _from_db_query:
                continue  # Will use niche score instead
            try:
                _vec = vec_func(_v.title_clean or "")
                _dist = float(np.linalg.norm(_vec.toarray() - local_centroid))
                _raw_dists[id(_v)] = _dist
            except Exception:
                pass

    _sim_scores = {}
    if _raw_dists:
        _min_d = min(_raw_dists.values())
        _max_d = max(_raw_dists.values())
        _d_range = _max_d - _min_d
        for _vid_key, _dist in _raw_dists.items():
            if _d_range > 0.001:
                # Closest video → 0.98, farthest → 0.50
                _norm = (_dist - _min_d) / _d_range
                _sim_scores[_vid_key] = round(0.98 - _norm * 0.48, 2)
            else:
                # All equidistant — use inverse distance, scaled to [0.50, 0.98]
                _sim_scores[_vid_key] = round(min(0.98, 1.0 / (1.0 + _dist)), 2)

    # Assign rank-based similarity scores for DB-queried videos.
    # Pure niche score saturates at 1.0 for all top gaming videos → all get 0.98.
    # Instead, interpolate 0.98 → 0.52 based on rank (already sorted best-first
    # by _video_rank_score = niche 60% + views 40%), giving meaningful differentiation.
    _niche_sim_scores = {}
    if _from_db_query:
        _db_vids_ordered = [v for v in top_cluster_vids if id(v) in _from_db_query]
        _n_db = len(_db_vids_ordered)
        for _rank, _v in enumerate(_db_vids_ordered):
            if _n_db > 1:
                _interp = _rank / (_n_db - 1)  # 0.0 for best, 1.0 for worst
                _niche_sim_scores[id(_v)] = round(0.98 - _interp * 0.46, 2)
            else:
                _niche_sim_scores[id(_v)] = 0.92  # single video

    for v in top_cluster_vids:
        vid_id = v.video_id or str(v.id)

        if id(v) in _niche_sim_scores:
            # DB-queried video — use niche relevance as similarity proxy
            sim_score = _niche_sim_scores[id(v)]
            method = "Niche Relevance Score"
        elif id(v) in _sim_scores:
            sim_score = _sim_scores[id(v)]
            method = "Distance to Centroid"
        else:
            sim_score = 0.85
            method = selection_method

        suggested_videos.append(
            {
                "videoId": vid_id,
                "title": v.title,
                "channel": v.channel_title or v.channel_id,
                "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/default.jpg",
                "views": v.view_count,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "similarityScore": sim_score,
                "selectionMethod": method,
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
        "analysisStartDate": start_date.isoformat(),
        "analysisEndDate": end_date.isoformat(),
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
    
    strategy_points = gemini_result.get("strategyPoints", [])
    core_trend_explanation = gemini_result.get("coreTrendExplanation", "")
    marketing_hooks = gemini_result.get("marketingHooks", [])

    # Fallback to top_terms if hooks are empty
    final_insights = marketing_hooks if marketing_hooks else top_terms[:5]

    strategy_engine = ml_signals.get("strategyEngine", "gemini")
    strategy_source = f"ml-topic-model-{strategy_engine}"

    # Confidence tier for response
    from ..ml.niche_filter import get_confidence_tier
    _final_conf_tier = get_confidence_tier(avg_confidence)

    # Both modes get videoIdeas and descriptionKeywords from Gemini
    video_ideas = gemini_result.get("videoIdeas", [])
    description_keywords = gemini_result.get("descriptionKeywords", [])

    response = {
        "region": region_norm,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "coreTrend": core_trend_label,
        "coreTrendExplanation": core_trend_explanation,
        "coreGenre": core_genre,
        "strategyPoints": strategy_points,
        "sentiment": sentiment,
        "dataVolume": data_volume,
        "relevanceScore": relevance_score,
        "predictionConfidence": avg_confidence,
        "confidenceTier": _final_conf_tier,
        "confidenceBreakdown": _confidence_breakdown,
        "trendMetrics": trend_metrics,
        "suggestedVideos": suggested_videos,
        "marketingInsights": final_insights,
        "videoIdeas": video_ideas,
        "descriptionKeywords": description_keywords,
        "mlSignals": ml_signals,
        "strategySource": strategy_source,
    }

    # ── Spread personalized Gemini fields into response ──
    if ml_signals.get("isPersonalized"):
        response["personalizedIntro"] = gemini_result.get("personalizedIntro")
        response["detailedStrategy"] = gemini_result.get("detailedStrategy")
        response["emergingTrends"] = gemini_result.get("emergingTrends", [])
        response["contentGaps"] = gemini_result.get("contentGaps", [])
        response["titleSuggestions"] = gemini_result.get("titleSuggestions", [])
        response["contentTone"] = gemini_result.get("contentTone", {})
        response["channelSpecificTips"] = gemini_result.get("channelSpecificTips", [])
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

    # Fallback uses 3 short punchy bullet points
    kw_str = ", ".join(top_keywords[:3]) if top_keywords else "relevant keywords"
    strategy_points = [
        f"{core_genre} content is expected to perform strongly in {region} — lead with a clear, keyword-rich title.",
        f"Focus on topics like {kw_str}. Comparison videos, hands-on demos, and step-by-step guides work best.",
        "Optimise thumbnails with a bold result or before-and-after to maximise click-through rate.",
    ]
    core_trend_explanation = (
        f"Creators producing {core_genre} content around {top_keywords[0] if top_keywords else 'trending topics'} "
        f"are seeing consistent audience growth in this region."
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
        "coreTrendExplanation": core_trend_explanation,
        "coreGenre": core_genre,
        "strategyPoints": strategy_points,
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