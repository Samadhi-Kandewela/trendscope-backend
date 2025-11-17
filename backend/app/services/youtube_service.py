from typing import List, Dict, Any, Optional # Added Any for the new function
import requests
from flask import current_app
# Added json for safe dummy data handling
import json 
# Added Any for type hinting in the new function
from typing import Dict, List, Any 


YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"


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


def get_keyword_analysis(seed_keyword: str, region: str) -> Dict[str, Any]:
    """
    Simulates fetching comprehensive keyword, tag, question, and competitive 
    analysis data for the Explorer page (Real-Time Tactical SEO).
    """
    # NOTE: This function simulates calling several external tools (YouTube Search API, 
    # SEO data providers) to match the required UI structure, as the YouTube 
    # Data API alone cannot provide all of this data (e.g., search volume, competition).
    
    region_display = region if region != "Global" else "Global"
    
    # --- 1. High-Volume Keywords (Simulation) ---
    high_volume_keywords = [
        {"keyword": "midjourney vs dall-e 2026", "volume": "29K", "competition": "Low", "trend_velocity": "Very High"},
        {"keyword": "best AI tools for creators", "volume": "45K", "competition": "Low", "trend_velocity": "High"},
        {"keyword": "free video editing AI", "volume": "38K", "competition": "Medium", "trend_velocity": "High"},
        {"keyword": "youtube automation with AI", "volume": "18K", "competition": "Medium", "trend_velocity": "High"},
        {"keyword": "AI scriptwriting tools", "volume": "22K", "competition": "Low", "trend_velocity": "Medium"},
    ]

    # --- 2. Trending Tags (Simulation) ---
    trending_tags = [
        "#viral videos", "#AI productivity", "#creator economy", 
        "#video marketing", "#social media trends", "#future of content"
    ]
    
    # --- 3. Audience Questions (Simulation) ---
    audience_questions = [
        "How will AI change video editing in 2026?",
        "What is the best free AI tool for scriptwriting?",
        "Is Midjourney still worth it in 2026?",
        "Top 5 new AI tools released this quarter",
        "How to use AI for YouTube SEO?",
    ]
    
    # --- 4. Competitive Analysis (Simulation) ---
    competitive_videos = [
        {
            "title": "I Tested 10 New AI Tools & Found 3 Must-Haves",
            "channel": "Tech Channel Pro",
            "views": "5.1M",
            "summary": "Review of free & paid tools focusing on workflow integration and time-saving features.",
            "tags": ["ai tools", "productivity apps", "must have 2026", "🔥 viral"],
            "insight": "Published 5 days ago. High Velocity — ideal for 'early adopter' audience.",
            "videoId": "abc123xyz" # Placeholder
        },
        {
            "title": "The End of Copywriting? Using GPT-5 for Scripts",
            "channel": "Frontend Masters",
            "views": "890K",
            "summary": "Breakdown of how LLMs replace junior copywriters for scriptwriting.",
            "tags": ["copywriting", "ai", "gpt-5", "script writing", "llm"],
            "insight": "Published 3 weeks ago. Strong long-tail performance with stable engagement.",
            "videoId": "def456uvw"
        },
        {
            "title": "AI Content Creation: Complete Beginner’s Guide 2026",
            "channel": "Content Academy",
            "views": "2.1M",
            "summary": "Step-by-step tutorial on automating video ideas and production using new AI systems.",
            "tags": ["AI content", "tutorial", "beginner guide", "2026 tools"],
            "insight": "Published 2 months ago. High authority, ranks consistently for broad keywords.",
            "videoId": "ghi789rst"
        },
    ]

    # The region input is intentionally ignored in the simulation logic above 
    # but would be used to filter real search data.
    return {
        "seed_keyword": seed_keyword,
        "region": region_display,
        "high_volume_keywords": high_volume_keywords,
        "trending_tags": trending_tags,
        "audience_questions": audience_questions,
        "competitive_videos": competitive_videos,
    }