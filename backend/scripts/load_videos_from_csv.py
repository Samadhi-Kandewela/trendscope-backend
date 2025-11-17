"""
Load filtered_2020_english.csv and filtered_old_dataset_english.csv
into the Postgres `videos` table.


"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from app import create_app
from app.extensions import db
from app.models.video import Video
from app.utils.genre_mapping import CATEGORY_TO_GENRE

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_2020 = BASE_DIR / "filtered_2020_english.csv"
CSV_OLD = BASE_DIR / "filtered_old_dataset_english.csv"

# Build reverse mapping: "Sports" -> 17, "Music" -> 10, etc.
CATEGORY_NAME_TO_ID = {name: cid for cid, name in CATEGORY_TO_GENRE.items()}


def parse_datetime(value) -> Optional[datetime]:
    """Convert CSV datetime field to timezone-aware pandas Timestamp (or None)."""
    if pd.isna(value):
        return None
    return pd.to_datetime(value, utc=True)


def parse_int(value) -> Optional[int]:
    """Safely cast view_count/like_count/... to int or return None."""
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_category_id(value) -> Optional[int]:
    """
    Normalize category field:

    - if numeric → int
    - if numeric string like "17" → int
    - if name like "Sports" → map via CATEGORY_NAME_TO_ID
    - else → None (stored as NULL)
    """
    if pd.isna(value):
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    if isinstance(value, str):
        value = value.strip()

        # Numeric string
        if value.isdigit():
            return int(value)

        # Category name
        if value in CATEGORY_NAME_TO_ID:
            return CATEGORY_NAME_TO_ID[value]

        # Unknown string
        print(f"[WARN] Unknown category value: {value!r} -> storing as NULL")
        return None

    return None


def load_2020(app):
    """Load filtered_2020_english.csv into `videos`."""
    if not CSV_2020.exists():
        print(f"{CSV_2020} not found, skipping.")
        return

    df = pd.read_csv(CSV_2020)
    print(f"Loading {len(df)} rows from {CSV_2020.name}")

    with app.app_context():
        for _, row in df.iterrows():
            tags = row.get("video_tags")
            tag_list = tags.split("|") if isinstance(tags, str) else []

            video = Video(
                id=row["video_id"],
                channel_id=row.get("channel_id"),
                channel_title=row.get("channel_title"),
                category_id=parse_category_id(row.get("video_category_id")),
                title=row.get("video_title"),
                description=row.get("video_description", ""),
                default_thumbnail=row.get("video_default_thumbnail", ""),
                tags=tag_list,
                view_count=parse_int(row.get("video_view_count")),
                like_count=parse_int(row.get("video_like_count")),
                dislike_count=None,
                comment_count=parse_int(row.get("video_comment_count")),
                trending_country=row.get("video_trending_country"),
                trending_date=parse_datetime(row.get("video_trending__date")),
                published_at=parse_datetime(row.get("video_published_at")),
                source_dataset="2020",
            )
            db.session.merge(video)  # upsert

        db.session.commit()
        print("2020 dataset loaded.")


def load_old(app):
    """Load filtered_old_dataset_english.csv into `videos`."""
    if not CSV_OLD.exists():
        print(f"{CSV_OLD} not found, skipping.")
        return

    df = pd.read_csv(CSV_OLD)
    print(f"Loading {len(df)} rows from {CSV_OLD.name}")

    with app.app_context():
        for _, row in df.iterrows():
            tags = row.get("video_tags")
            tag_list = tags.split("|") if isinstance(tags, str) else []

            video = Video(
                id=row["video_id"],
                channel_id=None,
                channel_title=row.get("channel_title"),
                category_id=parse_category_id(row.get("category_id")),
                title=row.get("video_title"),
                description=row.get("video_description", ""),
                default_thumbnail=row.get("video_default_thumbnail", ""),
                tags=tag_list,
                view_count=parse_int(row.get("video_view_count")),
                like_count=parse_int(row.get("video_like_count")),
                dislike_count=parse_int(row.get("video_dislike_count")),
                comment_count=parse_int(row.get("video_comment_count")),
                trending_country=row.get("video_trending_country"),
                trending_date=parse_datetime(row.get("video_trending__date")),
                published_at=parse_datetime(row.get("video_published_at")),
                source_dataset="old",
            )
            db.session.merge(video)

        db.session.commit()
        print("Old dataset loaded.")


if __name__ == "__main__":
    app = create_app()
    load_2020(app)
    load_old(app)
