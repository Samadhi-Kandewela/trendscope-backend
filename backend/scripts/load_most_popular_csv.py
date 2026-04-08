"""
Load most_popular_filtered_preprossesing.csv into the `videos` table.

Strategy:
- 4M rows but only ~150k unique video_ids (same video repeated across collection dates).
- Deduplicate to 1 row per video_id keeping the highest view_count record.
- Tags are space-separated text (not pipe-separated like other datasets).
- source_dataset = "most_popular"

Run from backend/ as:
    python -m scripts.load_most_popular_csv
"""

from pathlib import Path
from datetime import timezone
from typing import Optional

import pandas as pd

from app import create_app
from app.extensions import db
from app.models.video import Video

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "most_popular_filtered_preprossesing.csv"

CHUNK_SIZE = 200_000


def parse_datetime(value) -> Optional[object]:
    if pd.isna(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def parse_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_category_id(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def parse_tags(value) -> list:
    """
    most_popular tags are space-separated keywords stored as a single string.
    Store as a single-element list so tags_text downstream works correctly for TF-IDF.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    return [value.strip()]


def load_most_popular(app):
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    print(f"Reading {CSV_PATH.name} in chunks of {CHUNK_SIZE:,}...")

    # --- Pass 1: deduplicate across all chunks, keep highest view_count per video_id ---
    best: dict[str, dict] = {}

    for chunk_num, chunk in enumerate(
        pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    ):
        for _, row in chunk.iterrows():
            vid = row.get("video_id")
            if not vid or pd.isna(vid):
                continue

            vc = parse_int(row.get("view_count")) or 0
            existing = best.get(vid)
            if existing is None or vc > existing["view_count"]:
                best[vid] = {
                    "video_id":        vid,
                    "channel_id":      row.get("channel_id"),
                    "channel_title":   row.get("channel_title"),
                    "category_id":     row.get("category_id"),
                    "title":           row.get("title"),
                    "description":     row.get("description", ""),
                    "tags":            row.get("tags", ""),
                    "view_count":      vc,
                    "like_count":      parse_int(row.get("like_count")),
                    "comment_count":   parse_int(row.get("comment_count")),
                    "trending_country": row.get("region_code"),
                    "trending_date":   row.get("collection_date"),
                    "published_at":    row.get("published_at"),
                }

        processed = (chunk_num + 1) * CHUNK_SIZE
        print(f"  Scanned ~{min(processed, 4_026_876):,} rows, unique videos so far: {len(best):,}")

    print(f"\nDeduplication complete. {len(best):,} unique video_ids to insert.")

    # --- Pass 2: insert into videos table ---
    with app.app_context():
        inserted = 0
        batch = []

        for data in best.values():
            video = Video(
                id=data["video_id"],
                channel_id=str(data["channel_id"]) if data["channel_id"] and not pd.isna(data["channel_id"]) else None,
                channel_title=str(data["channel_title"]) if data["channel_title"] and not pd.isna(data["channel_title"]) else None,
                category_id=parse_category_id(data["category_id"]),
                title=str(data["title"]) if data["title"] and not pd.isna(data["title"]) else None,
                description=str(data["description"]) if data["description"] and not pd.isna(data["description"]) else "",
                default_thumbnail=None,
                tags=parse_tags(data["tags"]),
                view_count=data["view_count"] or None,
                like_count=data["like_count"],
                dislike_count=None,
                comment_count=data["comment_count"],
                trending_country=str(data["trending_country"]) if data["trending_country"] and not pd.isna(data["trending_country"]) else None,
                trending_date=parse_datetime(data["trending_date"]),
                published_at=parse_datetime(data["published_at"]),
                source_dataset="most_popular",
            )
            batch.append(video)

            if len(batch) >= 2000:
                for v in batch:
                    db.session.merge(v)
                db.session.commit()
                inserted += len(batch)
                batch = []
                print(f"  Inserted {inserted:,} / {len(best):,}")

        if batch:
            for v in batch:
                db.session.merge(v)
            db.session.commit()
            inserted += len(batch)

        print(f"\nDone. {inserted:,} rows inserted into videos (source_dataset='most_popular').")


if __name__ == "__main__":
    app = create_app()
    load_most_popular(app)
