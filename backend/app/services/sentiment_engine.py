import os
import json
from typing import List, Dict, Tuple
from flask import current_app
from google import genai
from google.genai.errors import APIError

_GEMINI_CLIENT = None
GEMINI_MODEL = "gemini-2.5-flash"

def _get_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            current_app.logger.error("GEMINI_API_KEY missing.")
            return None
        try:
            _GEMINI_CLIENT = genai.Client(api_key=api_key)
        except Exception as e:
            current_app.logger.error(f"Gemini init failed: {e}")
            return None
    return _GEMINI_CLIENT

def analyze_comments(comments: List[str]) -> Tuple[float, str]:
    """
    Analyzes a batch of comments (up to 50) and returns:
    (sentiment_score, dominant_emotion)
    
    sentiment_score: -1.0 (Negative) to 1.0 (Positive)
    dominant_emotion: e.g. "Excitement", "Anger", "Curiosity", "Skepticism"
    """
    if not comments:
        return 0.0, "Neutral"

    client = _get_client()
    if not client:
        return 0.0, "Service Unavailable"

    # Truncate to avoid massive context
    truncated_comments = [c[:200] for c in comments[:50]]
    text_block = "\n".join(f"- {c}" for c in truncated_comments)

    prompt = f"""
    Analyze the sentiment of these YouTube comments:
    
    {text_block}
    
    Output a JSON object with:
    1. "score": a float between -1.0 (negative) and 1.0 (positive).
    2. "emotion": a single word describing the dominant audience emotion.
    
    JSON only, no markdown.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        
        result = json.loads(response.text)
        score = float(result.get("score", 0.0))
        emotion = result.get("emotion", "Neutral")
        
        return score, emotion

    except Exception as e:
        current_app.logger.error(f"Sentiment analysis failed: {e}")
        return 0.0, "Error"
