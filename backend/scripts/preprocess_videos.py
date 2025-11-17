"""
Preprocess raw `videos` into a clean, ML-ready `videos_clean` table.

Run from backend/ as:

    python -m scripts.preprocess_videos
"""

from pathlib import Path
from typing import Optional, List

import pandas as pd
import re

from app import create_app
from app.extensions import db
from app.models.video import Video
from app.models.clean_video import CleanVideo
from app.utils.genre_mapping import CATEGORY_TO_UI_GENRE, CATEGORY_TO_GENRE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --- Country normalization mapping ---

COUNTRY_TO_ISO = {
    "United States": "US",
    "USA": "US",
    "US": "US",
    "Great Britain": "GB",
    "United Kingdom": "GB",
    "UK": "GB",
    "England": "GB",
    "India": "IN",
    "Canada": "CA",
    "Australia": "AU",
    "New Zealand": "NZ",
    "Germany": "DE",
    "France": "FR",
    "Italy": "IT",
    "Spain": "ES",
    "Japan": "JP",
    "South Korea": "KR",
    "Korea": "KR",
    # fallback: if already 2 letters, we'll uppercase it
}

def safe_bigint(value):
    """
    Safely convert a numeric-like value to Python int or None.

    - NaN / None → None
    - floats / strings → int(value) if possible
    - on any error → None
    """
    if value is None:
        return None
    try:
        import math
        # pandas NaN or float('nan') check
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass

    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None

def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    country = country.strip()
    if country in COUNTRY_TO_ISO:
        return COUNTRY_TO_ISO[country]
    if len(country) == 2:
        return country.upper()
    return None  # unknown or long-form we don't map yet


# --- Text cleaning helpers ---

URL_PATTERN = re.compile(r"https?://\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """
    Remove URLs, naive HTML tags, collapse whitespace.
    Does NOT lower-case or stem – we keep it readable for strategies.
    """
    if not text:
        return ""
    t = URL_PATTERN.sub(" ", text)
    t = HTML_TAG_PATTERN.sub(" ", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = MULTI_SPACE_PATTERN.sub(" ", t)
    return t.strip()


def build_dataframe(app) -> pd.DataFrame:
    """
    Pull all rows from `videos` into a pandas DataFrame with only the
    columns we care about (present in both CSV sources).
    """
    with app.app_context():
        videos: List[Video] = Video.query.all()

    rows = []
    for v in videos:
        tag_list = v.tags or []
        tags_text = " ".join(tag_list)

        rows.append({
            "video_id": v.id,
            "channel_id": v.channel_id,
            "channel_title": v.channel_title,
            "category_id": v.category_id,
            "title": v.title,
            "description": v.description,
            "tags_list": tag_list,
            "tags_text": tags_text,
            "trending_country_raw": v.trending_country,
            "trending_date": v.trending_date,
            "published_at": v.published_at,
            "view_count": v.view_count,
            "like_count": v.like_count,
            "dislike_count": v.dislike_count,
            "comment_count": v.comment_count,
            "source_dataset": v.source_dataset,
        })

    df = pd.DataFrame(rows)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates defined as same (video_id, trending_date).

    If multiple rows share those keys, keep the one with highest view_count.
    """
    # Ensure trending_date is datetime
    df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce")

    df = df.sort_values(
        ["video_id", "trending_date", "view_count"],
        ascending=[True, True, False],
    )
    df = df.drop_duplicates(subset=["video_id", "trending_date"], keep="first")
    return df


def drop_empty_title_or_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where title is empty OR tags_text is empty.
    This uses only columns that exist for BOTH datasets.
    """
    df["title"] = df["title"].fillna("").astype(str)
    df["tags_text"] = df["tags_text"].fillna("").astype(str)

    mask_valid = (df["title"].str.strip() != "") & (df["tags_text"].str.strip() != "")
    return df.loc[mask_valid].copy()


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean title, description, tags_text into *_clean columns.
    """
    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["tags_clean"] = df["tags_text"].apply(clean_text)
    return df


def fill_missing_category_with_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a simple text classifier (TF-IDF + LogisticRegression)
    on rows with known category_id, then predict category_id for
    rows with missing category_id based on title_clean + tags_clean.
    """
    # 1) Build training set
    df_train = df[df["category_id"].notna()].copy()
    df_missing = df[df["category_id"].isna()].copy()

    if df_missing.empty or df_train.empty:
        # Nothing to fill or no training data
        return df

    # Only keep rows with some text
    df_train["combined_text"] = (
        df_train["title_clean"].fillna("") + " " + df_train["tags_clean"].fillna("")
    )
    df_missing["combined_text"] = (
        df_missing["title_clean"].fillna("") + " " + df_missing["tags_clean"].fillna("")
    )

    df_train = df_train[df_train["combined_text"].str.strip() != ""]
    df_missing = df_missing[df_missing["combined_text"].str.strip() != ""]

    if df_train.empty or df_missing.empty:
        return df

    X_train_texts = df_train["combined_text"].tolist()
    y_train = df_train["category_id"].astype(int).tolist()

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_texts)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    X_missing = vectorizer.transform(df_missing["combined_text"].tolist())
    y_pred = clf.predict(X_missing)

    df.loc[df_missing.index, "category_id"] = y_pred
    return df


def add_genre_and_country_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Map category_id → UI genre using CATEGORY_TO_UI_GENRE
    - Normalize trending_country_raw → trending_country_code (ISO-2)
    """
    def map_genre(cid):
        if pd.isna(cid):
            return "Other"
        try:
            cid_int = int(cid)
        except (ValueError, TypeError):
            return "Other"
        return CATEGORY_TO_UI_GENRE.get(cid_int, "Other")

    df["genre"] = df["category_id"].apply(map_genre)
    df["trending_country_code"] = df["trending_country_raw"].apply(normalize_country)
    return df


def write_to_clean_table(app, df: pd.DataFrame):
    """
    Truncate `videos_clean` and bulk-insert cleaned rows.
    """
    with app.app_context():
        # Clear old data
        db.session.query(CleanVideo).delete()
        db.session.commit()

        # Insert cleaned rows
        objs = []
        for _, row in df.iterrows():
            cv = CleanVideo(
    video_id=row["video_id"],
    channel_id=row["channel_id"],
    channel_title=row["channel_title"],
    category_id=int(row["category_id"]) if not pd.isna(row["category_id"]) else None,
    genre=row["genre"],
    title=row["title"],
    title_clean=row["title_clean"],
    tags=row["tags_list"],
    tags_text=row["tags_text"],
    tags_clean=row["tags_clean"],
    description_clean=row["description_clean"],
    trending_country_raw=row["trending_country_raw"],
    trending_country_code=row["trending_country_code"],
    trending_date=row["trending_date"],
    published_at=row["published_at"],
    view_count=safe_bigint(row["view_count"]),
    like_count=safe_bigint(row["like_count"]),
    dislike_count=safe_bigint(row["dislike_count"]),
    comment_count=safe_bigint(row["comment_count"]),
    source_dataset=row["source_dataset"],
)

            objs.append(cv)

            # optional: flush in batches to save memory
            if len(objs) >= 2000:
                db.session.bulk_save_objects(objs)
                db.session.commit()
                objs = []

        if objs:
            db.session.bulk_save_objects(objs)
            db.session.commit()

        print(f"Inserted {len(df)} cleaned rows into videos_clean.")


def main():
    app = create_app()

    print("Building raw DataFrame from videos table...")
    df = build_dataframe(app)
    print(f"Raw rows: {len(df)}")

    print("Removing duplicates (video_id + trending_date)...")
    df = remove_duplicates(df)
    print(f"After dedup: {len(df)}")

    print("Dropping rows with empty title or tags...")
    df = drop_empty_title_or_tags(df)
    print(f"After dropping empty title/tags: {len(df)}")

    print("Cleaning text fields...")
    df = clean_text_fields(df)

    print("Filling missing category_id with NLP classifier...")
    df = fill_missing_category_with_nlp(df)

    print("Adding genre and normalized country codes...")
    df = add_genre_and_country_code(df)

    print("Writing to videos_clean table...")
    write_to_clean_table(app, df)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
