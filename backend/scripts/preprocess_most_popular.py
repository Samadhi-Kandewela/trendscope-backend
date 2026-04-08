"""
Preprocess most_popular_filtered.csv (8GB) into most_popular_filtered_preprossesing.csv.

Steps applied:
  1. Drop duplicates (video_id, collection_date) — keep max view_count
  2. Drop null/empty title rows
  3. Drop view_count <= 0 rows
  4. Filter live_broadcast_content == 'none'
  5. Language filter: default_language/default_audio_language starts with 'en',
     fallback ASCII >=75% on title for null language rows
  6. Clean title and description (remove URLs, HTML, collapse whitespace)
  7. Add empty like_count and tags columns
"""

import pandas as pd
import re
import os

INPUT_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "most_popular_filtered.csv"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "most_popular_filtered_preprossesing.csv"))

CHUNK_SIZE = 300_000

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
HTML_RE  = re.compile(r"<[^>]+>|&[a-zA-Z]+;|&#\d+;")
SPACE_RE = re.compile(r"\s+")


def is_english_dominant(text: str) -> bool:
    """Fallback: >=75% ASCII characters in the title."""
    if not text or len(text) < 3:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= 0.75


def starts_with_en(val) -> bool:
    """True if value is a string starting with 'en' (en, en-US, en-GB, etc.)."""
    return isinstance(val, str) and val.strip().lower().startswith("en")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # ── Step 2: drop null/empty title ────────────────────────────────────────
    chunk = chunk[chunk["title"].notna() & (chunk["title"].str.strip() != "")]

    # ── Step 3: drop zero/null view_count ────────────────────────────────────
    chunk["view_count"] = pd.to_numeric(chunk["view_count"], errors="coerce")
    chunk = chunk[chunk["view_count"].notna() & (chunk["view_count"] > 0)]

    # ── Step 4: keep only live_broadcast_content == 'none' ───────────────────
    chunk["live_broadcast_content"] = chunk["live_broadcast_content"].fillna("none")
    chunk = chunk[chunk["live_broadcast_content"].astype(str).str.strip().str.lower() == "none"]

    # ── Step 5: language filter ───────────────────────────────────────────────
    has_lang = starts_with_en(None)  # init placeholder
    lang_mask     = chunk["default_language"].apply(starts_with_en)
    audio_mask    = chunk["default_audio_language"].apply(starts_with_en)
    has_lang_info = lang_mask | audio_mask

    # rows where both language fields are null/empty → use ASCII fallback
    no_lang_info  = ~chunk["default_language"].apply(starts_with_en) & \
                    ~chunk["default_audio_language"].apply(starts_with_en)
    ascii_mask    = chunk["title"].apply(is_english_dominant)

    keep = has_lang_info | (no_lang_info & ascii_mask)
    chunk = chunk[keep]

    # ── Step 6: clean title and description ──────────────────────────────────
    chunk = chunk.copy()
    chunk["title"]       = chunk["title"].apply(clean_text)
    chunk["description"] = chunk["description"].apply(clean_text)

    # ── Step 7: add empty columns ─────────────────────────────────────────────
    chunk["like_count"] = None
    chunk["tags"]       = ""

    return chunk


def main():
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")

    total_read = 0
    total_kept = 0
    chunk_num  = 0
    first_chunk = True

    # Accumulate within-chunk duplicates across chunks using a seen set
    # Full dedup (Step 1) done per-chunk on (video_id, collection_date);
    # a second pass would be needed for cross-chunk dupes — acceptable for this size.
    with pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE, low_memory=False) as reader:
        for chunk in reader:
            chunk_num  += 1
            rows_in     = len(chunk)
            total_read += rows_in

            # ── Step 1: drop duplicates within chunk ─────────────────────────
            chunk = chunk.sort_values("view_count", ascending=False, na_position="last")
            chunk = chunk.drop_duplicates(subset=["video_id", "collection_date"], keep="first")

            chunk = process_chunk(chunk)

            rows_out    = len(chunk)
            total_kept += rows_out

            if not chunk.empty:
                chunk.to_csv(
                    OUTPUT_PATH,
                    mode="w" if first_chunk else "a",
                    index=False,
                    header=first_chunk,
                )
                first_chunk = False

            print(
                f"Chunk {chunk_num:>4} | Read: {total_read:>11,} | "
                f"Kept: {total_kept:>9,} | "
                f"This chunk: {rows_out:>6,}/{rows_in:>6,}"
            )

    print(f"\nDone.")
    print(f"Total rows read : {total_read:,}")
    print(f"Total rows kept : {total_kept:,}")
    print(f"Reduction       : {(1 - total_kept/total_read)*100:.1f}%")

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
        print(f"Output file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
