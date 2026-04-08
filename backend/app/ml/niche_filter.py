"""
Niche Relevance Filter
======================
Ensures trend recommendations stay within the creator's declared content genre.
Prevents off-niche viral content (music videos, cat videos) from contaminating
trend signals for niche creators (travel, gaming, etc.).

Phase 3 fixes applied:
  1. filter_videos_by_niche: threshold raised 0.1 -> 0.33 (was letting almost everything through)
  2. filter_videos_by_niche: min_keep fallback now never includes score=0.0 videos
     (was forcing 200 off-niche videos in regardless of relevance)
  3. GENRE_BLACKLIST added: hard-blocks cross-contamination keywords per genre
     (e.g. Gaming queries will never return makeup/skincare videos)
  4. compute_video_niche_score: caps score at 0.05 if blacklisted terms found
  5. filter_keywords_by_niche: preserve_count fallback now only keeps keywords
     with score > 0.0 (was preserving completely off-niche keywords)
"""

from typing import List, Dict, Optional


# -----------------------------------------------------------------------
# DB Genre Mapping: Maps creator profile genre labels → actual DB genre values
# (CleanVideo.genre column uses different labels than creator profile fields)
# -----------------------------------------------------------------------
DB_GENRE_MAP: Dict[str, List[str]] = {
    "Travel":      ["Vlogs", "Lifestyle"],
    "Beauty":      ["Lifestyle"],
    "Fitness":     ["Lifestyle"],
    "Finance":     ["Lifestyle", "Education"],
    "Food":        ["Lifestyle", "Vlogs"],
    "Technology":  ["Tech"],
    "Gaming":      ["Gaming"],
    "Music":       ["Music"],
    "Sports":      ["Sports"],
    "News":        ["News"],
    "Education":   ["Education"],
    "Entertainment": ["Lifestyle", "Vlogs"],
    "Lifestyle":   ["Lifestyle", "Vlogs"],
    # DB genres map to themselves
    "Vlogs":       ["Vlogs"],
    "Tech":        ["Tech"],
    "Other":       ["Other"],
}


# -----------------------------------------------------------------------
# Genre Taxonomy: Maps UI genre labels to relevant anchor keywords
# -----------------------------------------------------------------------
NICHE_TAXONOMY: Dict[str, List[str]] = {
    "Travel": [
        "travel", "vlog", "destination", "trip", "tour", "flight", "hotel",
        "backpacking", "culture", "adventure", "explore", "city", "country",
        "abroad", "vacation", "holiday", "road", "beach", "mountain", "island",
        "itinerary", "budget", "hostel", "airbnb", "passport", "visa",
        "luggage", "airport", "cruise", "world", "local", "guide", "tourist",
        "wanderlust", "journey", "expedition", "globe", "trotter", "nomad",
    ],
    "Gaming": [
        "game", "gaming", "gameplay", "walkthrough", "playthrough", "stream",
        "esports", "fps", "rpg", "mmorpg", "review", "ps5", "xbox", "nintendo",
        "switch", "pc", "steam", "twitch", "raid", "build", "speedrun", "tips",
        "strategy", "level", "boss", "loot", "character", "mod", "patch",
        "console", "controller", "multiplayer", "pvp", "minecraft", "fortnite",
        # Mobile gaming & battle royale
        "clash", "royale", "brawl", "stars", "arena", "clan", "wars", "quest",
        "gems", "coins", "trophy", "tournament", "rank", "ranked", "season",
        "mobile", "android", "ios", "app", "supercell", "epic", "comeback",
        "challenge", "battle", "squad", "victory", "defeat", "mission",
        # General gaming actions
        "kill", "win", "lose", "score", "round", "match", "league", "pro",
        "noob", "grind", "farm", "op", "nerf", "buff", "update", "new",
        "trailer", "reveal", "reaction", "impression", "call", "duty", "ops",
    ],
    "Technology": [
        "tech", "technology", "review", "unboxing", "laptop", "phone", "android",
        "iphone", "app", "software", "code", "programming", "ai", "robot",
        "gadget", "setup", "pc", "computer", "update", "release", "specs",
        "benchmark", "comparison", "tutorial", "developer", "hardware", "cpu",
        "gpu", "monitor", "keyboard", "mouse", "cable", "charger", "battery",
    ],
    "Beauty": [
        "makeup", "beauty", "skincare", "tutorial", "routine", "foundation",
        "lipstick", "eyeshadow", "hair", "styling", "product", "review",
        "glam", "glow", "skin", "tone", "palette", "brush", "contour", "blush",
        "serum", "moisturizer", "cleanser", "toner", "sunscreen", "anti-aging",
        "haul", "drugstore", "luxury", "dupe", "swatch", "lashes", "brow",
    ],
    "Food": [
        "food", "recipe", "cook", "cooking", "baking", "restaurant", "eat",
        "meal", "dish", "cuisine", "kitchen", "chef", "taste", "delicious",
        "healthy", "snack", "dessert", "breakfast", "lunch", "dinner", "vegan",
        "vegetarian", "gluten", "keto", "paleo", "diet", "nutrition", "calorie",
        "grocery", "ingredient", "sauce", "seasoning", "grilling", "roasting",
    ],
    "Fitness": [
        "workout", "fitness", "exercise", "gym", "weight", "muscle", "cardio",
        "yoga", "pilates", "running", "training", "health", "diet", "nutrition",
        "strength", "body", "transformation", "challenge", "routine", "sport",
        "calories", "protein", "supplements", "stretching", "hiit", "crossfit",
        "marathon", "cycling", "swimming", "abs", "squat", "deadlift", "bench",
    ],
    "Education": [
        "learn", "tutorial", "how", "explained", "course", "study", "school",
        "university", "subject", "math", "science", "history", "language",
        "english", "lesson", "class", "teach", "knowledge", "tips", "guide",
        "lecture", "concept", "theory", "practice", "exam", "quiz", "homework",
        "research", "experiment", "discovery", "facts", "documentary",
    ],
    "Entertainment": [
        "funny", "comedy", "reaction", "challenge", "prank", "viral", "meme",
        "compilation", "fail", "best", "top", "rank", "celebrity", "celebrity",
        "award", "show", "movie", "film", "dance", "talent", "skit", "parody",
        "sketch", "stand-up", "impressions", "roast", "blind", "audition",
    ],
    "Lifestyle": [
        "life", "vlog", "day", "routine", "morning", "night", "self", "care",
        "productivity", "motivation", "minimalist", "home", "decor", "fashion",
        "outfit", "style", "personal", "growth", "mindset", "wellness", "mood",
        "journal", "planner", "organize", "clean", "apartment", "budget",
        "relationship", "dating", "marriage", "family", "parenting",
    ],
    "Finance": [
        "money", "finance", "investing", "crypto", "stocks", "budget", "saving",
        "income", "wealth", "financial", "passive", "side", "hustle", "rich",
        "bank", "loan", "tax", "retire", "portfolio", "market", "trading",
        "dividend", "etf", "reit", "real estate", "401k", "ira", "compound",
        "debt", "frugal", "millionaire", "billion", "startup", "business",
    ],
    "Music": [
        "music", "song", "album", "official", "video", "mv", "lyrics", "cover",
        "remix", "artist", "band", "release", "single", "concert", "live",
        "performance", "beat", "rap", "pop", "rnb", "kpop", "jazz", "classical",
        "acoustic", "instrumental", "playlist", "spotify", "producer", "studio",
    ],
    "Sports": [
        "sport", "football", "soccer", "basketball", "cricket", "tennis",
        "golf", "nfl", "nba", "mlb", "fifa", "match", "game", "team",
        "player", "goal", "highlights", "championship", "tournament", "coach",
        "transfer", "draft", "season", "playoffs", "finals", "injury", "trade",
    ],
    "News": [
        "news", "breaking", "update", "report", "world", "politics", "economy",
        "social", "event", "crisis", "war", "election", "government", "policy",
        "analysis", "editorial", "interview", "press", "media", "broadcast",
    ],
}

# -----------------------------------------------------------------------
# Genre Blacklist: Keywords that definitively prove a video is OFF-NICHE
# for a given genre. If any blacklisted term appears in the video text,
# the niche score is capped at 0.05 regardless of other matches.
#
# Purpose: Prevent false positives where off-genre videos share one
# incidental word with the niche vocabulary (e.g., "game" in a beauty
# video title, or "foundation" in a gaming video).
# -----------------------------------------------------------------------
GENRE_BLACKLIST: Dict[str, List[str]] = {
    "Gaming": [
        "makeup", "skincare", "lipstick", "foundation", "eyeshadow",
        "mascara", "blush", "contour", "serum", "moisturizer", "cleanser",
        "haircare", "hair extensions", "nail", "lashes", "brow",
        "recipe", "cooking", "baking", "restaurant", "cuisine",
        "investing", "stocks", "crypto", "dividend", "portfolio",
        # Late-night TV / talk show content — definitively not gaming
        "tonight show", "jimmy fallon", "late night", "talk show",
        "fallon", "nbc studios", "american idol", "the voice",
        "saturday night live", "snl", "late show", "daily show",
    ],
    "Beauty": [
        "fps", "raid", "loot", "esports", "speedrun", "boss fight",
        "multiplayer", "pvp", "mmorpg", "walkthrough", "playthrough",
        "minecraft", "fortnite", "roblox", "xbox", "ps5", "nintendo",
        "investing", "stocks", "crypto", "nfl", "nba", "cricket",
    ],
    "Food": [
        "esports", "fps", "gameplay", "walkthrough", "raid", "loot",
        "investing", "stocks", "crypto", "trading", "portfolio",
        "makeup", "skincare", "foundation", "eyeshadow",
        "nfl", "nba", "cricket", "football highlights",
    ],
    "Finance": [
        "makeup", "skincare", "foundation", "eyeshadow", "lipstick",
        "gameplay", "esports", "fps", "walkthrough", "minecraft",
        "recipe", "cooking", "baking", "restaurant",
        "nfl", "nba", "cricket", "football highlights",
    ],
    "Technology": [
        "makeup", "skincare", "foundation", "eyeshadow", "lipstick",
        "recipe", "cooking", "baking", "restaurant", "cuisine",
        "esports", "fps", "walkthrough", "raid", "loot",
        "nfl", "nba", "cricket", "football highlights",
    ],
    "Travel": [
        "makeup tutorial", "skincare routine", "foundation", "eyeshadow",
        "esports", "fps", "gameplay", "raid", "loot", "minecraft",
        "investing", "stocks", "crypto", "trading",
        "nfl", "nba", "cricket", "wrestling",
    ],
    "Fitness": [
        "makeup", "skincare", "foundation", "eyeshadow",
        "esports", "fps", "gameplay", "walkthrough", "minecraft",
        "investing", "stocks", "crypto", "trading",
        "recipe", "restaurant", "cuisine",
    ],
    "Education": [
        "makeup tutorial", "skincare routine",
        "esports", "fps", "raid", "loot", "speedrun",
        "investing tips", "crypto trading",
    ],
    "Sports": [
        "makeup", "skincare", "foundation", "eyeshadow",
        "investing", "stocks", "crypto", "trading",
        "recipe", "cooking", "baking", "restaurant",
        "minecraft", "roblox", "fortnite", "esports fps",
    ],
    "Music": [
        "makeup tutorial", "skincare routine", "foundation",
        "esports", "fps", "raid", "speedrun",
        "investing", "stocks", "crypto",
        "recipe", "cooking", "baking",
        "football highlights", "nba highlights",
    ],
    "Entertainment": [],  # Entertainment is broad — no blacklist
    "Lifestyle": [],       # Lifestyle is broad — no blacklist
    "News": [],            # News is broad — no blacklist
}


# Cross-genre keywords that should never be filtered (universal relevance)
UNIVERSAL_KEYWORDS = {
    "viral", "trending", "best", "top", "review", "tutorial", "tips",
    "guide", "challenge", "2025", "2026", "how", "watch", "new", "latest",
}

# Keywords to always exclude regardless of niche (pure platform noise)
NOISE_KEYWORDS = {
    "youtube", "subscribe", "like", "comment", "share", "channel", "video",
    "notification", "bell", "official",
}


def get_niche_vocabulary(genre: str) -> List[str]:
    """Returns the anchor vocabulary for a given genre (case-insensitive)."""
    if not genre:
        return []
    vocab = NICHE_TAXONOMY.get(genre)
    if vocab:
        return vocab
    for key, val in NICHE_TAXONOMY.items():
        if key.lower() == genre.lower():
            return val
    return []


def compute_keyword_niche_score(keyword: str, genre: str) -> float:
    """
    Returns a relevance score 0.0–1.0 for how niche-relevant a keyword is.

    Scoring:
    - 1.0: Exact match in niche vocabulary
    - 0.7: Partial match (keyword contains or is contained in niche term)
    - 0.5: Universal keyword (cross-genre relevance)
    - 0.0: Off-niche noise
    """
    if not keyword or not genre:
        return 0.5

    kw_lower = keyword.lower().strip()

    if kw_lower in NOISE_KEYWORDS:
        return 0.0

    if kw_lower in UNIVERSAL_KEYWORDS:
        return 0.5

    niche_vocab = get_niche_vocabulary(genre)
    if not niche_vocab:
        return 0.5  # Unknown genre — neutral

    if kw_lower in niche_vocab:
        return 1.0

    for niche_term in niche_vocab:
        if kw_lower in niche_term or niche_term in kw_lower:
            return 0.7

    return 0.0


def filter_keywords_by_niche(
    keywords: List[str],
    genre: str,
    threshold: float = 0.3,
    preserve_count: int = 5,
) -> List[str]:
    """
    Filters a keyword list to only include niche-relevant keywords.

    Phase 3 fix: preserve_count fallback now only keeps keywords with
    score > 0.0. The old code preserved the top-N keywords regardless
    of score, meaning completely off-niche keywords (score=0.0) were
    preserved as a fallback. Now if fewer than preserve_count keywords
    have score > 0.0, we return only those that do.

    Returns: Filtered keyword list ordered by relevance score desc.
    """
    if not genre or not keywords:
        return keywords

    scored = [(kw, compute_keyword_niche_score(kw, genre)) for kw in keywords]
    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [kw for kw, score in scored if score >= threshold]

    if len(filtered) < preserve_count:
        # Only fall back to keywords with score > 0.0 — never include off-niche (score=0.0)
        filtered = [kw for kw, score in scored[:preserve_count] if score > 0.0]

    return filtered


def compute_video_niche_score(video, genre: str) -> float:
    """
    Scores how niche-relevant a single video is to the given genre.
    Checks title + tags + description for niche vocabulary density.

    Phase 3 fix: GENRE_BLACKLIST check added.
    If any blacklisted term is found in the video text, score is capped
    at 0.05 regardless of niche matches. This prevents false positives
    where a beauty video mentions "game" once and passes a Gaming filter.

    Returns: 0.0–1.0
    """
    if not genre:
        return 0.5

    niche_vocab = get_niche_vocabulary(genre)
    if not niche_vocab:
        return 0.5

    title = (getattr(video, 'title_clean', None) or getattr(video, 'title', '') or '').lower()
    tags = (getattr(video, 'tags_clean', None) or getattr(video, 'tags_text', '') or '').lower()
    desc = (getattr(video, 'description_clean', '') or '')[:300].lower()

    full_text = f"{title} {tags} {desc}"
    if not full_text.strip():
        return 0.0

    # Check blacklist before scoring — any blacklisted term kills relevance
    blacklist = GENRE_BLACKLIST.get(genre, [])
    if blacklist:
        for blocked_term in blacklist:
            if blocked_term in full_text:
                return 0.05  # Hard cap — definitively off-niche

    matches = sum(1 for term in niche_vocab if term in full_text)
    # 3+ niche term matches = fully relevant
    score = min(1.0, matches / 3.0)
    return round(score, 2)


def filter_videos_by_niche(
    videos: list,
    genre: str,
    threshold: float = 0.33,
    min_keep: int = 200,
) -> list:
    """
    Pre-filters a video pool to niche-relevant content before clustering.

    Keeps videos with niche_score >= threshold.

    Phase 3 fixes:
      - threshold raised from 0.1 to 0.33: a video needs at least 1 solid
        niche term match to pass (old 0.1 let virtually everything through)
      - min_keep fallback fixed: if fewer than min_keep videos pass, relax
        threshold by 0.1 and retry once, then take top results with score > 0.0.
        Never include score=0.0 (completely off-niche) videos.
        Old code forced top-200 regardless of score, meaning off-niche videos
        were always included when data was sparse.
    """
    if not genre or genre.lower() in ("", "global", "all"):
        return videos

    niche_vocab = get_niche_vocabulary(genre)
    if not niche_vocab:
        return videos  # Unknown genre — skip filtering

    scored = [(v, compute_video_niche_score(v, genre)) for v in videos]
    filtered = [(v, s) for v, s in scored if s >= threshold]

    if len(filtered) < min_keep:
        # First retry: relax threshold by 0.1 and try again
        relaxed_threshold = max(0.0, threshold - 0.1)
        filtered = [(v, s) for v, s in scored if s >= relaxed_threshold]

    if len(filtered) < min_keep:
        # Still not enough — take top min_keep but ONLY those with score > 0.0
        # Never include completely off-niche videos (score=0.0)
        non_zero = [(v, s) for v, s in scored if s > 0.0]
        filtered = sorted(non_zero, key=lambda x: x[1], reverse=True)[:min_keep]

    return [v for v, _ in filtered]


def get_confidence_tier(confidence: float) -> Dict:
    """
    Maps a raw confidence float (0.0–1.0) to a human-readable tier.
    Used to calibrate how the system presents its recommendations.
    """
    if confidence >= 0.7:
        return {
            "tier": "high",
            "label": "Strong Signal",
            "warningMessage": None,
            "suggestedAction": None,
        }
    elif confidence >= 0.4:
        return {
            "tier": "medium",
            "label": "Moderate Signal",
            "warningMessage": (
                "Trend signals are moderate for this period. "
                "Recommendations are directionally sound but should be validated."
            ),
            "suggestedAction": (
                "Monitor results over 2–3 weeks before committing to a full strategy shift."
            ),
        }
    else:
        return {
            "tier": "low",
            "label": "Weak Signal",
            "warningMessage": (
                "Trend signals are weak for this date range — possibly sparse data "
                "or too broad a date range. Treat these as exploratory suggestions, "
                "not confirmed strategies."
            ),
            "suggestedAction": (
                "Test 1–2 content pieces before pivoting. "
                "Consider narrowing your date range (e.g., 30 days) for stronger signals."
            ),
        }
