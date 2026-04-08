"""
Filter most_popular.csv (78GB) to only keep the 13 countries used in TrendScope.
Uses chunked reading to avoid loading the full file into memory.
"""

import pandas as pd
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "most_popular.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "most_popular_filtered.csv")

KEEP_COUNTRIES = {
    "US", "GB", "IN", "CA", "AU",  # Primary (scheduler)
    "NZ", "IE",                      # English-speaking filter
    "LK", "PH", "SG", "ZA", "MY",  # Extended region aliases
}

CHUNK_SIZE = 500_000

def main():
    input_path = os.path.abspath(INPUT_PATH)
    output_path = os.path.abspath(OUTPUT_PATH)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Countries: {sorted(KEEP_COUNTRIES)}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")

    total_read = 0
    total_kept = 0
    chunk_num = 0
    first_chunk = True

    with pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False) as reader:
        for chunk in reader:
            chunk_num += 1
            rows_in = len(chunk)
            total_read += rows_in

            filtered = chunk[chunk["region_code"].isin(KEEP_COUNTRIES)]
            rows_out = len(filtered)
            total_kept += rows_out

            if not filtered.empty:
                filtered.to_csv(
                    output_path,
                    mode="w" if first_chunk else "a",
                    index=False,
                    header=first_chunk,
                )
                first_chunk = False

            print(
                f"Chunk {chunk_num:>4} | Read: {total_read:>12,} | "
                f"Kept: {total_kept:>10,} | "
                f"This chunk: {rows_out:>7,}/{rows_in:>7,}"
            )

    print(f"\nDone.")
    print(f"Total rows read : {total_read:,}")
    print(f"Total rows kept : {total_kept:,}")
    print(f"Filter rate     : {total_kept/total_read*100:.1f}%")

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"Output file size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
