"""
Merge tags from tag.csv (50GB) into most_popular_filtered_preprossesing.csv.

Strategy (memory-safe):
  Pass 1 — Load join keys from preprocessed file into a set (~300MB RAM).
  Pass 2 — Stream tag.csv in 500K-row chunks, keep only rows matching our keys,
            accumulate into a dict: {(collection_date, region_code, rank) -> [tags]}.
  Pass 3 — Read preprocessed CSV in chunks, fill tags column from dict, write output.

Output overwrites most_popular_filtered_preprossesing.csv in-place via a temp file.
"""

import pandas as pd
import os
import tempfile
from collections import defaultdict

PREPROCESSED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "most_popular_filtered_preprossesing.csv")
)
TAGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tag.csv")
)
TEMP_PATH = PREPROCESSED_PATH + ".tmp"

CHUNK_SIZE = 500_000


def make_key(row_date, row_region, row_rank):
    """Normalise key to (date_str, region_str, rank_int)."""
    return (str(row_date).strip(), str(row_region).strip(), int(row_rank))


# ── PASS 1: build key set from preprocessed file ─────────────────────────────
print("Pass 1 — loading join keys from preprocessed file...")
key_set = set()

for chunk in pd.read_csv(PREPROCESSED_PATH, usecols=["collection_date", "region_code", "rank"],
                          chunksize=CHUNK_SIZE, low_memory=False):
    for row in chunk.itertuples(index=False):
        try:
            key_set.add(make_key(row.collection_date, row.region_code, row.rank))
        except (ValueError, TypeError):
            pass

print(f"  Unique keys loaded: {len(key_set):,}\n")


# ── PASS 2: stream tag.csv, accumulate matching tags ─────────────────────────
print("Pass 2 — streaming tag.csv and collecting matching tags...")
tags_dict = defaultdict(list)

total_tag_rows  = 0
matched_tag_rows = 0
chunk_num = 0

with pd.read_csv(TAGS_PATH, chunksize=CHUNK_SIZE, low_memory=False) as reader:
    for chunk in reader:
        chunk_num      += 1
        total_tag_rows += len(chunk)

        # filter to only rows whose key exists in our preprocessed set
        chunk = chunk.dropna(subset=["collection_date", "region_code", "rank", "tag"])
        chunk["rank"] = pd.to_numeric(chunk["rank"], errors="coerce").dropna()
        chunk = chunk.dropna(subset=["rank"])
        chunk["rank"] = chunk["rank"].astype(int)

        for row in chunk.itertuples(index=False):
            key = (str(row.collection_date).strip(), str(row.region_code).strip(), int(row.rank))
            if key in key_set:
                tag = str(row.tag).strip()
                if tag:
                    tags_dict[key].append(tag)
                    matched_tag_rows += 1

        print(
            f"  Chunk {chunk_num:>4} | Tag rows read: {total_tag_rows:>12,} | "
            f"Matched so far: {matched_tag_rows:>9,} | Dict size: {len(tags_dict):>7,}"
        )

# collapse list of tags → single space-separated string
print(f"\nCollapsing tag lists into strings...")
tags_merged = {k: " ".join(v) for k, v in tags_dict.items()}
del tags_dict  # free memory
print(f"  Videos with tags: {len(tags_merged):,}\n")


# ── PASS 3: write preprocessed file with tags filled in ──────────────────────
print("Pass 3 — merging tags into preprocessed file...")

total_rows  = 0
tagged_rows = 0
first_chunk = True

with pd.read_csv(PREPROCESSED_PATH, chunksize=CHUNK_SIZE, low_memory=False) as reader:
    for chunk in reader:
        total_rows += len(chunk)

        chunk["rank"] = pd.to_numeric(chunk["rank"], errors="coerce").fillna(0).astype(int)

        def fill_tags(row):
            key = (str(row["collection_date"]).strip(), str(row["region_code"]).strip(), int(row["rank"]))
            return tags_merged.get(key, "")

        chunk["tags"] = chunk.apply(fill_tags, axis=1)
        tagged_rows  += (chunk["tags"] != "").sum()

        chunk.to_csv(
            TEMP_PATH,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )
        first_chunk = False

        print(f"  Rows processed: {total_rows:>9,} | Tagged so far: {tagged_rows:>9,}")

# replace original with updated file
os.replace(TEMP_PATH, PREPROCESSED_PATH)

print(f"\nDone.")
print(f"Total rows     : {total_rows:,}")
print(f"Rows with tags : {tagged_rows:,}  ({tagged_rows/total_rows*100:.1f}%)")
size_mb = os.path.getsize(PREPROCESSED_PATH) / (1024 ** 2)
print(f"Output file    : {size_mb:.1f} MB  →  {PREPROCESSED_PATH}")
