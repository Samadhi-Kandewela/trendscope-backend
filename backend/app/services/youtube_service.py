from typing import List, Dict, Any, Optional # Added Any for the new function
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
_GEMINI_CLIENT_INSTANCE = None

GENRE_TO_CATEGORY_ID = {
    "Gaming": 20,
    "Music": 10,
    "Tech": 28,      # Science & Technology
    "Education": 27,
    "Lifestyle": 26, # Howto & Style
    "Vlogs": 22,     # People & Blogs
}

# --- YOUTUBE DATA API CLIENT GETTER (for external use) ---
def _get_gemini_client():
    """Lazily initializes the Gemini client."""
    global _GEMINI_CLIENT_INSTANCE
    if _GEMINI_CLIENT_INSTANCE is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            current_app.logger.error("FATAL: GEMINI_API_KEY environment variable is missing.")
            return None
        try:
            _GEMINI_CLIENT_INSTANCE = genai.Client(api_key=api_key)
        except Exception as e:
            current_app.logger.error("Gemini Client initialization failed: %s", str(e))
    return _GEMINI_CLIENT_INSTANCE

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

def _enrich_video_data(youtube_key: str, video_ids: List[str]) -> Dict[str, Dict]:
    """
    Makes the second API call (Batch Request) to get statistics and tags for the videos.
    Returns a dictionary mapping videoId to its stats/tags.
    """
    if not video_ids:
        return {}

    id_string = ",".join(video_ids)
    
    params = {
        "key": youtube_key,
        "id": id_string,
        "part": "statistics,snippet", # Request statistics AND snippet (which contains tags)
        "maxResults": 50
    }
    
    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/videos", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        enriched_data = {}
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
        current_app.logger.error(f"YouTube Enrichment API failed: {e}")
        return {}

def _fetch_competitive_videos(youtube_key: str, seed_keyword: str, region: str) -> List[Dict]:
    """
    STAGE 1: Search videos and get IDs.
    """
    if not youtube_key:
        return []

    # ... (params setup remains the same)

    params = {
        "key": youtube_key,
        "q": seed_keyword,
        "part": "snippet",
        "type": "video",
        "regionCode": region,
        "maxResults": 3,
        "order": "viewCount" # Use viewCount to get the most successful videos
    }

    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/search", params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        search_results: List[Dict] = []
        video_ids: List[str] = []
        
        for item in data.get("items", []):
            vid_id = item["id"].get("videoId")
            if vid_id:
                video_ids.append(vid_id)
                search_results.append({
                    "title": item["snippet"].get("title"),
                    "channel": item["snippet"].get("channelTitle"),
                    "videoId": vid_id,
                    "summary": item["snippet"].get("description", "No description available.").split('.')[0],
                    "thumbnail": item["snippet"].get("thumbnails", {}).get("default", {}).get("url"),
                })

        # STAGE 2: Enrichment (calls the new function)
        enriched_data = _enrich_video_data(youtube_key, video_ids)
        
        final_videos = []
        for video in search_results:
            video_id = video['videoId']
            enrichment = enriched_data.get(video_id, {})
            
            # --- MERGE DATA ---
            views = enrichment.get("views", "N/A")
            tags = enrichment.get("tags", ["N/A"])
            
            # Create a simple insight based on the available data
            insight_text = f"Views: {_format_volume(views)}. Ranks highly for this search query."
                
            video['views'] = _format_volume(views) # Format views for UI
            video['tags'] = tags[:4] # Use real tags, truncated for clean display
            video['insight'] = insight_text
            
            # Add a basic summary for the competitive insight panel
            video['summary'] = video['summary'] if len(video['summary']) < 150 else video['summary'][:150] + "..."
            
            final_videos.append(video)
            
        return final_videos
        
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"YouTube Search API failed: {e}")
        return []
    
def _generate_audience_questions(seed_keyword: str, high_volume_keywords: List[Dict]) -> List[str]:
    """
    Uses Gemini to generate smart, niche-specific audience questions based on the 
    user's keyword and the generated keyword list.
    """
    client = _get_gemini_client()
    if client is None:
        return ["AI Question Generator Unavailable.", "Check service configuration."]
        
    top_terms = [k['keyword'] for k in high_volume_keywords]

    prompt = f"""
    You are a YouTube SEO strategist and niche expert. Your goal is to generate 5 highly valuable, click-worthy video title ideas phrased as questions.
    
    The audience and strategic focus must be inferred from the primary topic: '{seed_keyword}' and the top search terms: {', '.join(top_terms)}.

    Instructions:
    1. Infer the target audience (e.g., Gamers, Investors, Developers, Consumers).
    2. Generate questions that address the audience's biggest current challenges, fears, or aspirations related to the topic.
    3. Ensure at least two questions incorporate a specific keyword from the provided list.
    4. Generate 5 distinct questions and format the output as a simple Python list of strings. Do not include any prefix, numbers, or introduction.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0.6,
            ),
        )
        # Attempt to parse the response text as a Python list (LLMs often wrap it in code blocks)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split('\n', 1)[1].rstrip("`")
        
        # Safely evaluate the string as a list of strings
        questions = json.loads(text)
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            return questions
        
        # Fallback if parsing fails
        return [f"Generated questions for '{seed_keyword}' failed to parse.", "Try searching another term."]

    except Exception as e:
        current_app.logger.error(f"Gemini Question generation error: {e}")
        return [f"AI Service Error: {str(e)[:50]}", "Using default generic questions.", "What are the biggest challenges in this niche?"]

def get_keyword_analysis(seed_keyword: str, region: str) -> Dict[str, Any]:
    """
    Fetches real keyword analysis data by calling both RapidAPI (metrics) 
    and YouTube Data API (competitive videos) and Gemini (questions).
    """
    rapidapi_key = current_app.config.get("RAPIDAPI_KEY")
    youtube_key = current_app.config.get("YOUTUBE_API_KEY")

    if not rapidapi_key:
        return _handle_api_failure(seed_keyword, region, "RAPIDAPI_KEY not configured.")

    region_display = region if region != "Global" else "Global"
    
    # --- 1. RAPIDAPI CALL (Metrics) ---
    # ... (RapidAPI call and error handling remain the same)
    
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


    # --- 3. YOUTUBE API CALL (Competitive Videos - ENRICHMENT) ---
    competitive_videos = _fetch_competitive_videos(youtube_key, seed_keyword, region)
    
    
    # --- 4. GEMINI API CALL (Audience Questions - REPLACES STATIC LIST) ---
    audience_questions = _generate_audience_questions(seed_keyword, high_volume_keywords)
    
    
    # --- 5. Return Final Structure ---
    return {
        "seed_keyword": seed_keyword,
        "region": region_display,
        "high_volume_keywords": high_volume_keywords[:5],
        "trending_tags": trending_tags[:6],
        "audience_questions": audience_questions,
        "competitive_videos": competitive_videos,
    }