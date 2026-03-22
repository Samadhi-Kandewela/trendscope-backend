from typing import List, Dict, Any, Optional
import requests
import json
import os
from datetime import datetime
from flask import current_app
from google import genai
from google.genai.errors import APIError

# --- API CONFIGURATION ---
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
RAPIDAPI_HOST = "youtube-keywords-in-google-trends.p.rapidapi.com"
RAPIDAPI_ENDPOINT = f"https://{RAPIDAPI_HOST}"

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
                "channelId": snippet.get("channelId"),
                "thumbnail": (
                    snippet.get("thumbnails", {})
                    .get("medium", {})
                    .get("url")
                ),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "commentCount": int(stats.get("commentCount", 0)),
                "publishedAt": snippet.get("publishedAt"),
                "categoryId": snippet.get("categoryId"),
                "description": description,
                "tags": snippet.get("tags", []),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            }
        )

    return videos


def fetch_all_trending_videos(region: str) -> List[Dict[str, Any]]:
    """
    Fetches trending videos for ALL tracked genres in a specific region.
    Used by the Scheduler for daily data ingestion.
    """
    all_videos = []
    
    # tracked genres from config or constant
    genres = ["Gaming", "Music", "Tech", "Education", "Lifestyle", "Vlogs"]
    
    for genre in genres:
        try:
            # We reuse the existing single-genre fetcher
            # limit=50 is standard max for YouTube API
            videos = get_trending_videos(region=region, genre=genre, limit=50)
            
            # Tag them with the genre for downstream processing if needed
            for v in videos:
                v["genre_label"] = genre
                v["region_code"] = region
                
            all_videos.extend(videos)
        except Exception as e:
            current_app.logger.error(f"Failed to fetch {genre} for {region}: {e}")
            continue
            
    return all_videos


# ---------------------------------------------------------------------------
# Channel info fetcher (for Creator Onboarding)
# ---------------------------------------------------------------------------
def fetch_channel_info(channel_url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches YouTube channel info from a URL.
    Supports: /@handle, /channel/UCxxxx, /c/customname, /user/username
    Returns dict with channelId, channelName, subscriberCount, etc.
    """
    import re

    youtube_key = os.getenv("YOUTUBE_API_KEY") or (
        current_app.config.get("YOUTUBE_API_KEY")
    )
    if not youtube_key:
        return {"error": "YouTube API key not configured"}

    channel_id = None
    # Pattern 1: /channel/UCxxxxxxx
    m = re.search(r"/channel/(UC[\w-]+)", channel_url)
    if m:
        channel_id = m.group(1)

    # Pattern 2: /@handle or /c/name or /user/name
    if not channel_id:
        handle = None
        m = re.search(r"/@([\w.-]+)", channel_url)
        if m:
            handle = m.group(1)
        else:
            m = re.search(r"/(?:c|user)/([\w.-]+)", channel_url)
            if m:
                handle = m.group(1)

        if handle:
            # Use search to resolve handle to channel ID
            try:
                search_url = f"{YOUTUBE_API_URL}/search"
                resp = requests.get(search_url, params={
                    "key": youtube_key,
                    "q": handle,
                    "type": "channel",
                    "part": "snippet",
                    "maxResults": 1,
                }, timeout=10)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    if items:
                        channel_id = items[0]["snippet"]["channelId"]
            except Exception as e:
                current_app.logger.error("Channel search failed: %s", e)

    if not channel_id:
        return {"error": "Could not extract channel ID from URL"}

    # Fetch full channel details
    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/channels", params={
            "key": youtube_key,
            "id": channel_id,
            "part": "snippet,statistics",
        }, timeout=10)

        if resp.status_code != 200:
            return {"error": f"YouTube API error: {resp.status_code}"}

        items = resp.json().get("items", [])
        if not items:
            return {"error": "Channel not found"}

        ch = items[0]
        snippet = ch.get("snippet", {})
        stats = ch.get("statistics", {})

        return {
            "channelId": channel_id,
            "channelName": snippet.get("title", ""),
            "description": (snippet.get("description") or "")[:500],
            "thumbnail": (
                snippet.get("thumbnails", {}).get("medium", {}).get("url")
                or snippet.get("thumbnails", {}).get("default", {}).get("url")
                or ""
            ),
            "subscriberCount": int(stats.get("subscriberCount", 0)),
            "totalViews": int(stats.get("viewCount", 0)),
            "videoCount": int(stats.get("videoCount", 0)),
        }

    except Exception as e:
        current_app.logger.error("fetch_channel_info failed: %s", e)
        return {"error": str(e)}


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
                "trend_velocity": "N/A",
            }
        ],
        "trending_tags": [f"{seed_keyword.replace(' ', '-').lower()}", "#DATA_ERROR"],
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
        current_app.logger.error("YOUTUBE_API_KEY not configured; cannot fetch competitive videos.")
        return []

    # Base params for YouTube Search API
    params: Dict[str, Any] = {
        "key": youtube_key,
        "q": seed_keyword,
        "part": "snippet",
        "type": "video",
        "maxResults": 3,
        "order": "viewCount",  # get most viewed
    }

    
    if region != "Global":
        params["regionCode"] = region  # e.g. "US", "GB", "IN", ...

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

        current_app.logger.info(
            "Fetched %d competitive videos for '%s' (region=%s)",
            len(final_videos),
            seed_keyword,
            region,
        )
        return final_videos

    except requests.exceptions.RequestException as e:
        current_app.logger.error(
            "YouTube Search API failed for '%s' (region=%s): %s",
            seed_keyword,
            region,
            e,
        )
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

    # Dynamically get the current year so we never allow older years
    current_year = datetime.utcnow().year

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
4. IMPORTANT: Do NOT invent or include any calendar year earlier than {current_year}.
   - If the seed keyword or top search terms contain a year (e.g. "2026"),
     you may use that year, but you must never introduce a year smaller than {current_year}.
5. If no year is explicitly provided in the seed keyword or top terms, either:
   - use {current_year} if a year really makes sense, OR
   - avoid mentioning any year at all.
6. Generate 5 distinct questions and format the output as a JSON list of
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
    Fetches keyword analysis data by calling:
      - RapidAPI (YouTube keyword suggestions + search counts)
      - YouTube Data API (competitive videos)
      - Gemini (audience questions)

    New RapidAPI used:
      youtube-keywords-in-google-trends.p.rapidapi.com

    Output:
      - high_volume_keywords: [
            { "keyword": str,
              "volume": str,        # formatted: "600", "1.2K", "3.4M"
              "trend_velocity": str # "High" / "Medium" / "Low" (relative to others)
            }, ...
        ]
      - trending_tags: ["emailmarketingtools", "emailmarketingjobs", ...]
      - competitive_videos: [...]   # unchanged (from YouTube API)
      - audience_questions: [...]   # unchanged (from Gemini)
    """
    rapidapi_key = current_app.config.get("RAPIDAPI_KEY")
    youtube_key = current_app.config.get("YOUTUBE_API_KEY")

    if not rapidapi_key:
        return _handle_api_failure(seed_keyword, region, "RAPIDAPI_KEY not configured.")

    # Display label – you can keep using the actual region code if you want
    region_display = region if region != "Global" else "Global"

    # --- 1. RAPIDAPI CALL (Metrics via youtube-keywords-in-google-trends) ---
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": rapidapi_key,
    }

    # This API expects keyword in the path, e.g. /Emailmarketing
    # We'll URL-encode the seed keyword to be safe.
    from requests.utils import quote

    url = f"{RAPIDAPI_ENDPOINT}/{quote(seed_keyword)}"

    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        api_data = response.json()
    except requests.exceptions.RequestException as e:
        current_app.logger.exception("RapidAPI request error")
        return _handle_api_failure(seed_keyword, region_display, str(e))

    # --- 2. Extract keyword suggestions from the weird nested structure ---
    keywords_raw: List[Dict[str, Any]] = []

    if isinstance(api_data, list):
        for block in api_data:
            # We only care about blocks that are lists with length > 1
            if isinstance(block, list) and len(block) > 1:
                # Skip the first element (label like "YouTube Suggestions" or "c")
                for item in block[1:]:
                    if (
                        isinstance(item, dict)
                        and "YouTube_Suggestion" in item
                        and "Search_count_in_YouTube" in item
                    ):
                        keywords_raw.append(item)

    if not keywords_raw:
        return _handle_api_failure(
            seed_keyword,
            region_display,
            "No keyword suggestions in API response.",
        )

    # --- 3. Rank-based Trend Velocity on TOP 5 only ---
    # Sort all suggestions by raw volume descending
    keywords_raw.sort(
        key=lambda k: int(k.get("Search_count_in_YouTube", 0) or 0),
        reverse=True,
    )

    # Take only top 5 items (these are what we will display anyway)
    top_items = keywords_raw[:5]

    # Extract raw volumes for those top items
    top_volumes: List[int] = []
    for item in top_items:
        try:
            v = int(item.get("Search_count_in_YouTube", 0) or 0)
        except (TypeError, ValueError):
            v = 0
        top_volumes.append(v)

    # Sort a copy of volumes in descending order to determine ranks
    # Keep duplicates so ties share the same rank
    sorted_vals = sorted(top_volumes, reverse=True)

    # Example: sorted_vals = [1200, 1000, 1000, 700, 300]
    # ranks = [1, 2, 2, 4, 5]
    ranks: List[int] = []
    for v in sorted_vals:
        ranks.append(sorted_vals.index(v) + 1)

    def classify_by_rank(volume: int) -> str:
        """
        Assign High / Medium / Low based on descending rank (with ties).
        - Top 2 ranks => High
        - Rank 3       => Medium
        - Rank 4 & 5   => Low
        Ties share the same rank (because of index() usage).
        """
        try:
            idx = sorted_vals.index(volume)
            rank = ranks[idx]
        except ValueError:
            # If somehow volume not found (shouldn't happen), treat as Low
            return "Low"

        if rank <= 2:
            return "High"
        if rank == 3:
            return "Medium"
        return "Low"

    # --- 4. Build high_volume_keywords + trending_tags ---
    high_volume_keywords: List[Dict[str, Any]] = []
    trending_tags: List[str] = []

    for item, volume_raw in zip(top_items, top_volumes):
        keyword = item.get("YouTube_Suggestion", "N/A")
        volume_str = _format_volume(volume_raw)
        trend_velocity = classify_by_rank(volume_raw)

        high_volume_keywords.append(
            {
                "keyword": keyword,
                "volume": volume_str,
                "trend_velocity": trend_velocity,
            }
        )

        # Build a clean tag version (no spaces / hyphens). We will NOT prepend '#'
        tag_name = keyword.replace(" ", "").replace("-", "").lower()
        if tag_name and len(trending_tags) < 10:
            trending_tags.append(tag_name)

    # --- 5. YOUTUBE API CALL (Competitive Videos - ENRICHMENT) ---
    competitive_videos = _fetch_competitive_videos(
        youtube_key=youtube_key,
        seed_keyword=seed_keyword,
        region=region,
    )

    # --- 6. GEMINI API CALL (Audience Questions) ---
    audience_questions = _generate_audience_questions(
        seed_keyword=seed_keyword,
        high_volume_keywords=high_volume_keywords,
    )

    # --- 7. Return Final Structure ---
    return {
        "seed_keyword": seed_keyword,
        "region": region_display,
        "high_volume_keywords": high_volume_keywords,
        "trending_tags": trending_tags[:6],
        "audience_questions": audience_questions,
        "competitive_videos": competitive_videos,
    }


def get_video_comments(video_id: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches top comments for a video.
    """
    api_key = current_app.config.get("YOUTUBE_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "videoId": video_id,
        "part": "snippet,id",
        "maxResults": max_results,
        "textFormat": "plainText",
        "order": "relevance",
    }

    try:
        # Use YOUTUBE_API_URL if defined, else hardcode base
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        comments = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append({
                "id": item.get("id"),
                "text": snippet.get("textDisplay"),
                "author": snippet.get("authorDisplayName"),
                "likes": snippet.get("likeCount", 0),
                "publishedAt": snippet.get("publishedAt")
            })
            
        return comments

    except Exception as e:
        current_app.logger.error(f"Failed to fetch comments for {video_id}: {e}")
        return []