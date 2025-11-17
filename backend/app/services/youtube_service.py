from typing import List, Dict, Any, Optional # Added Any for the new function
import requests
from flask import current_app
# Added json for safe dummy data handling
import json 

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

# --- RAPIDAPI CONFIGURATION ---
RAPIDAPI_HOST = "keyword-research-for-youtube.p.rapidapi.com"
RAPIDAPI_ENDPOINT = f"https://{RAPIDAPI_HOST}/yttags.php"

GENRE_TO_CATEGORY_ID = {
    "Gaming": 20,
    "Music": 10,
    "Tech": 28,      # Science & Technology
    "Education": 27,
    "Lifestyle": 26, # Howto & Style
    "Vlogs": 22,     # People & Blogs
}


def get_trending_videos(region: str, genre: str, limit: int = 8) -> List[Dict]:
    """
    Calls YouTube Data API to get chart=mostPopular videos,
    optionally filtered by category based on the UI 'genre'.
    
    This function supports the Dashboard page.
    """
    api_key = current_app.config.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY (YT_API_KEY) not configured")

    params = {
        "key": api_key,
        "chart": "mostPopular",
        "regionCode": region,
        "part": "snippet,statistics",
        "maxResults": min(limit, 50),
    }

    category_id = GENRE_TO_CATEGORY_ID.get(genre)
    if category_id:
        params["videoCategoryId"] = category_id

    resp = requests.get(f"{YOUTUBE_API_URL}/videos", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    videos: List[Dict] = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        vid_id = item["id"]

        videos.append(
            {
                "videoId": vid_id,
                "title": snippet.get("title"),
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

def _format_volume(volume: int | float) -> str:
    """Helper to convert raw volume numbers to K/M format."""
    if volume >= 1_000_000:
        return f"{round(volume / 1_000_000, 1)}M"
    if volume >= 1_000:
        return f"{round(volume / 1000)}K"
    return str(int(volume))

def _handle_api_failure(seed_keyword: str, region: str, error: str) -> Dict[str, Any]:
    """Provides a safe, minimal fallback structure when the API call fails."""
    current_app.logger.error(f"API call failed for '{seed_keyword}': {error}")
    return {
        "seed_keyword": seed_keyword,
        "region": region,
        "high_volume_keywords": [
            {"keyword": f"{seed_keyword} - Data Unavailable", "volume": "N/A", "competition": "N/A", "trend_velocity": "N/A"}
        ],
        "trending_tags": [f"#{seed_keyword.replace(' ', '-').lower()}", "#DATA_ERROR"],
        "audience_questions": [f"Data service failed: {error[:50]}", "Try searching a broader keyword."],
        "competitive_videos": [], 
    }

def _fetch_competitive_videos(youtube_key: str, seed_keyword: str, region: str) -> List[Dict]:
    """
    Fetches the actual top-ranking videos using the YouTube Data API Search endpoint.
    This replaces the simulation.
    """
    if not youtube_key:
        current_app.logger.warning("YouTube API key missing for competitive video search.")
        return []

    params = {
        "key": youtube_key,
        "q": seed_keyword,
        "part": "snippet",
        "type": "video",
        "regionCode": region,
        "maxResults": 3, # We only need the Top 3
        "order": "viewCount" # Order by view count to get high-ranking content
    }

    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/search", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        videos = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            vid_id = item["id"].get("videoId")

            videos.append(
                {
                    "title": snippet.get("title"),
                    "channel": snippet.get("channelTitle"),
                    "videoId": vid_id,
                    # We can't get views or tags without a second 'videos' API call, so we use N/A placeholders for views/tags
                    "views": "N/A (2nd Call Needed)", 
                    "summary": snippet.get("description", "No description available.").split('.')[0],
                    "tags": ["N/A", "N/A"],
                    "insight": "Ranks based on relevance/views.",
                }
            )
        return videos
        
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"YouTube Search API failed: {e}")
        return [] # Return empty list on failure

def get_keyword_analysis(seed_keyword: str, region: str) -> Dict[str, Any]:
    """
    Fetches real keyword analysis data by calling both RapidAPI (metrics) 
    and YouTube Data API (competitive videos).
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
        response = requests.get(RAPIDAPI_ENDPOINT, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        api_data = response.json()
    except requests.exceptions.RequestException as e:
        # Fallback if metrics fail
        return _handle_api_failure(seed_keyword, region, str(e))


    # --- 2. Process Metrics Data ---
    keywords_to_process = []
    if api_data.get('exact_keyword'):
        keywords_to_process.extend(api_data['exact_keyword'])
    if api_data.get('related_keywords'):
        keywords_to_process.extend(api_data['related_keywords'])

    high_volume_keywords = []
    trending_tags = []
    
    for item in keywords_to_process:
        keyword = item.get("keyword", "N/A")
        volume = item.get("monthlysearch", 0)
        competition_score = item.get("competition_score", 0)
        difficulty = item.get("difficulty", "N/A")

        high_volume_keywords.append({
            "keyword": keyword,
            "volume": _format_volume(volume),
            "competition": difficulty,
            "trend_velocity": "High" if competition_score < 40 else "Medium"
        })
        
        tag_name = keyword.replace(' ', '').replace('-', '').lower()
        if tag_name and len(trending_tags) < 10:
            trending_tags.append(f"#{tag_name}")


    # --- 3. YOUTUBE API CALL (Competitive Videos - Replaces Simulation) ---
    competitive_videos = _fetch_competitive_videos(youtube_key, seed_keyword, region)
    
    
    # --- 4. Return Final Structure ---
    return {
        "seed_keyword": seed_keyword,
        "region": region_display,
        "high_volume_keywords": high_volume_keywords[:5],
        "trending_tags": trending_tags[:6],
        "audience_questions": [
            f"Why is {high_volume_keywords[0]['keyword']} trending now?",
            "What type of video format works best for this niche?",
            "Should I focus on short-form content for this topic?",
            "Top 5 questions about " + high_volume_keywords[0]['keyword'].split()[0],
        ],
        "competitive_videos": competitive_videos,
    }