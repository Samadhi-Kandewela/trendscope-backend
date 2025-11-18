from typing import List, Dict, Any, Optional
import requests
import json
import os

from flask import current_app
from google import genai
from google.genai.errors import APIError

# --- API CONFIGURATION ---
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
RAPIDAPI_HOST = "keyword-research-for-youtube.p.rapidapi.com"
RAPIDAPI_ENDPOINT = f"https://{RAPIDAPI_HOST}/yttags.php"

GEMINI_MODEL = "gemini-2.5-flash"
_GEMINI_CLIENT_INSTANCE: Optional[genai.Client] = None

GENRE_TO_CATEGORY_ID = {
    "Gaming": 20,
    "Music": 10,
    "Tech": 28,        # Science & Technology
    "Education": 27,
    "Lifestyle": 26,   # Howto & Style
    "Vlogs": 22,       # People & Blogs
}

# Extra keywords to tighten the Education genre
EDU_KEYWORDS = [
    "study",
    "course",
    "class",
    "lesson",
    "tutorial",
    "learning",
    "learn",
    "exam",
    "test",
    "revision",
    "how to",
    "lecture",
    "math",
    "science",
    "physics",
    "chemistry",
    "biology",
    "english",
    "ielts",
    "ielts",
    "university",
    "school",
    "college",
]


# ---------------------------------------------------------------------------
# Gemini client helper
# ---------------------------------------------------------------------------
def _get_gemini_client() -> Optional[genai.Client]:
    """Lazily initializes the Gemini client."""
    global _GEMINI_CLIENT_INSTANCE

    if _GEMINI_CLIENT_INSTANCE is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            current_app.logger.error(
                "FATAL: GEMINI_API_KEY environment variable is missing."
            )
            return None
        try:
            _GEMINI_CLIENT_INSTANCE = genai.Client(api_key=api_key)
        except Exception as e:  # noqa: BLE001
            current_app.logger.error("Gemini Client initialization failed: %s", e)
            return None

    return _GEMINI_CLIENT_INSTANCE


# ---------------------------------------------------------------------------
# Simple heuristic for education videos
# ---------------------------------------------------------------------------
def _looks_educational(title: str, description: str) -> bool:
    """
    Heuristic filter for Education genre:
    Returns True if the title/description contains any word
    from EDU_KEYWORDS.
    """
    text = f"{title} {description}".lower()
    return any(keyword in text for keyword in EDU_KEYWORDS)


# ---------------------------------------------------------------------------
# Trending videos for dashboard
# ---------------------------------------------------------------------------
def get_trending_videos(region: str, genre: str, limit: int = 8) -> List[Dict]:
    """
    Calls YouTube Data API to get chart=mostPopular videos,
    optionally filtered by category based on the UI 'genre'.

    This function supports the Dashboard page.
    """
    api_key = current_app.config.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY (YT_API_KEY) not configured")

    params: Dict[str, Any] = {
        "key": api_key,
        "chart": "mostPopular",
        "part": "snippet,statistics",
        "maxResults": min(limit, 50),
    }

    # The YouTube API does not accept "Global" as a regionCode.
    # To get global trending videos, the regionCode parameter must be omitted.
    if region != "Global":
        params["regionCode"] = region

    category_id = GENRE_TO_CATEGORY_ID.get(genre)
    if category_id:
        params["videoCategoryId"] = category_id

    resp = requests.get(f"{YOUTUBE_API_URL}/videos", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    videos: List[Dict[str, Any]] = []

    for item in data.get("items", []):
        snippet = item.get("snippet", {}) or {}
        stats = item.get("statistics", {}) or {}
        vid_id = item.get("id")

        if not vid_id:
            continue

        title = snippet.get("title", "") or ""
        description = snippet.get("description", "") or ""

        # EXTRA FILTER: if genre is Education, skip videos that don't look educational
        if genre == "Education" and not _looks_educational(title, description):
            continue

        videos.append(
            {
                "videoId": vid_id,
                "title": title,
                "channel": snippet.get("channelTitle"),
                "thumbnail": (
                    snippet.get("thumbnails", {})
                    .get("medium", {})
                    .get("url")
                ),
                "views": int(stats.get("viewCount", 0)),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            }
        )

    return videos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_volume(volume: int | float) -> str:
    """Helper to convert raw volume numbers to K/M format."""
    if volume >= 1_000_000:
        return f"{round(volume / 1_000_000, 1)}M"
    if volume >= 1_000:
        return f"{round(volume / 1000)}K"
    return str(int(volume))


def _handle_api_failure(seed_keyword: str, region: str, error: str) -> Dict[str, Any]:
    """Provides a safe, minimal fallback structure when the API call fails."""
    current_app.logger.error("API call failed for '%s': %s", seed_keyword, error)
    return {
        "seed_keyword": seed_keyword,
        "region": region,
        "high_volume_keywords": [
            {
                "keyword": f"{seed_keyword} - Data Unavailable",
                "volume": "N/A",
                "competition": "N/A",
                "trend_velocity": "N/A",
            }
        ],
        "trending_tags": [f"#{seed_keyword.replace(' ', '-').lower()}", "#DATA_ERROR"],
        "audience_questions": [
            f"Data service failed: {error[:50]}",
            "Try searching a broader keyword.",
        ],
        "competitive_videos": [],
    }


# ---------------------------------------------------------------------------
# YouTube enrichment helpers
# ---------------------------------------------------------------------------
def _enrich_video_data(
    youtube_key: str, video_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Makes the second API call (Batch Request) to get statistics and tags
    for the videos. Returns a dictionary mapping videoId to its stats/tags.
    """
    if not video_ids:
        return {}

    id_string = ",".join(video_ids)

    params = {
        "key": youtube_key,
        "id": id_string,
        "part": "statistics,snippet",  # statistics AND snippet (which contains tags)
        "maxResults": 50,
    }

    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/videos", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        enriched_data: Dict[str, Dict[str, Any]] = {}
        for item in data.get("items", []):
            item_id = item["id"]
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})

            enriched_data[item_id] = {
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "tags": snippet.get("tags", []),
            }

        return enriched_data

    except requests.exceptions.RequestException as e:
        current_app.logger.error("YouTube Enrichment API failed: %s", e)
        return {}


def _fetch_competitive_videos(
    youtube_key: Optional[str],
    seed_keyword: str,
    region: str,
) -> List[Dict[str, Any]]:
    """
    Fetches competitive videos for the keyword.

    STAGE 1: Search videos and get IDs.
    STAGE 2: Enrich those IDs with statistics and tags.
    """
    if not youtube_key:
        return []

    params = {
        "key": youtube_key,
        "q": seed_keyword,
        "part": "snippet",
        "type": "video",
        "regionCode": region,
        "maxResults": 3,
        "order": "viewCount",  # Use viewCount to get the most successful videos
    }

    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/search", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        search_results: List[Dict[str, Any]] = []
        video_ids: List[str] = []

        for item in data.get("items", []):
            vid_id = item["id"].get("videoId")
            if vid_id:
                video_ids.append(vid_id)
                snippet = item["snippet"]
                description = snippet.get("description", "No description available.")
                first_sentence = description.split(".")[0]

                search_results.append(
                    {
                        "title": snippet.get("title"),
                        "channel": snippet.get("channelTitle"),
                        "videoId": vid_id,
                        "summary": first_sentence,
                        "thumbnail": (
                            snippet.get("thumbnails", {})
                            .get("default", {})
                            .get("url")
                        ),
                    }
                )

        # STAGE 2: Enrichment
        enriched_data = _enrich_video_data(youtube_key, video_ids)

        final_videos: List[Dict[str, Any]] = []
        for video in search_results:
            video_id = video["videoId"]
            enrichment = enriched_data.get(video_id, {})

            views = enrichment.get("views", 0)
            tags = enrichment.get("tags", ["N/A"])

            video["views"] = _format_volume(views)
            video["tags"] = tags[:4]

            insight_text = (
                f"Views: {video['views']}. Ranks highly for this search query."
            )
            video["insight"] = insight_text

            summary = video["summary"]
            if len(summary) > 150:
                summary = summary[:150] + "..."
            video["summary"] = summary

            final_videos.append(video)

        return final_videos

    except requests.exceptions.RequestException as e:
        current_app.logger.error("YouTube Search API failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Gemini-based audience question generation
# ---------------------------------------------------------------------------
def _generate_audience_questions(
    seed_keyword: str,
    high_volume_keywords: List[Dict[str, Any]],
) -> List[str]:
    """
    Uses Gemini to generate smart, niche-specific audience questions based on the
    user's keyword and the generated keyword list.
    """
    client = _get_gemini_client()
    if client is None:
        return [
            "AI Question Generator Unavailable.",
            "Check service configuration.",
        ]

    top_terms = [k["keyword"] for k in high_volume_keywords]

    prompt = f"""
You are a YouTube SEO strategist and niche expert. Your goal is to generate 5
highly valuable, click-worthy video title ideas phrased as questions.

The audience and strategic focus must be inferred from the primary topic:
'{seed_keyword}' and the top search terms: {', '.join(top_terms)}.

Instructions:
1. Infer the target audience (e.g., Gamers, Investors, Developers, Consumers).
2. Generate questions that address the audience's biggest current challenges,
   fears, or aspirations related to the topic.
3. Ensure at least two questions incorporate a specific keyword from the list.
4. Generate 5 distinct questions and format the output as a JSON list of
   strings. Do not include any prefix, numbers, or introduction.
""".strip()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0.6,
            ),
        )

        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # remove first line (``` or ```json) and trailing ```
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3].strip()

        # Try to parse as JSON list
        questions = json.loads(text)
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            return questions

        return [
            f"Generated questions for '{seed_keyword}' failed to parse.",
            "Try searching another term.",
        ]

    except Exception as e:  # noqa: BLE001
        current_app.logger.error("Gemini Question generation error: %s", e)
        return [
            f"AI Service Error: {str(e)[:50]}",
            "Using default generic questions.",
            "What are the biggest challenges in this niche?",
        ]


# ---------------------------------------------------------------------------
# Public API: Keyword analysis
# ---------------------------------------------------------------------------
def get_keyword_analysis(seed_keyword: str, region: str) -> Dict[str, Any]:
    """
    Fetches real keyword analysis data by calling:
      - RapidAPI (metrics)
      - YouTube Data API (competitive videos)
      - Gemini (audience questions)
    """
    rapidapi_key = current_app.config.get("RAPIDAPI_KEY")
    youtube_key = current_app.config.get("YOUTUBE_API_KEY")

    if not rapidapi_key:
        return _handle_api_failure(seed_keyword, region, "RAPIDAPI_KEY not configured.")

    region_display = region if region != "Global" else "Global"

    # --- 1. RAPIDAPI CALL (Metrics) ---
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": rapidapi_key,
    }
    params = {
        "keyword": seed_keyword,
        "country": region,
    }

    try:
        response = requests.get(
            RAPIDAPI_ENDPOINT,
            headers=headers,
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        api_data = response.json()
    except requests.exceptions.RequestException as e:
        return _handle_api_failure(seed_keyword, region, str(e))

    # --- 2. Process Metrics Data ---
    keywords_to_process: List[Dict[str, Any]] = []
    if api_data.get("exact_keyword"):
        keywords_to_process.extend(api_data["exact_keyword"])
    if api_data.get("related_keywords"):
        keywords_to_process.extend(api_data["related_keywords"])

    high_volume_keywords: List[Dict[str, Any]] = []
    trending_tags: List[str] = []

    for item in keywords_to_process:
        keyword = item.get("keyword", "N/A")
        volume = item.get("monthlysearch", 0)
        competition_score = item.get("competition_score", 0)
        difficulty = item.get("difficulty", "N/A")

        high_volume_keywords.append(
            {
                "keyword": keyword,
                "volume": _format_volume(volume),
                "competition": difficulty,
                "trend_velocity": "High" if competition_score < 40 else "Medium",
            }
        )

        tag_name = keyword.replace(" ", "").replace("-", "").lower()
        if tag_name and len(trending_tags) < 10:
            trending_tags.append(f"#{tag_name}")

    # --- 3. YOUTUBE API CALL (Competitive Videos - ENRICHMENT) ---
    competitive_videos = _fetch_competitive_videos(
        youtube_key=youtube_key,
        seed_keyword=seed_keyword,
        region=region,
    )

    # --- 4. GEMINI API CALL (Audience Questions) ---
    audience_questions = _generate_audience_questions(
        seed_keyword=seed_keyword,
        high_volume_keywords=high_volume_keywords,
    )

    # --- 5. Return Final Structure ---
    return {
        "seed_keyword": seed_keyword,
        "region": region_display,
        "high_volume_keywords": high_volume_keywords[:5],
        "trending_tags": trending_tags[:6],
        "audience_questions": audience_questions,
        "competitive_videos": competitive_videos,
    }
