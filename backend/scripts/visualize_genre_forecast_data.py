"""
Visualize the training data used for genre forecasting.

Usage (from backend folder):
    python -m scripts.visualize_genre_forecast_data
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from app import create_app
from app.extensions import db
from app.models.clean_video import CleanVideo


def build_region_month_genre_shares():
    """
    Build aggregated genre share data per (region, month).

    Returns:
        regions: sorted list of region codes
        genres: sorted list of genres
        shares: dict[(region, month)] -> dict[genre] = share (0–1)
    """
    rows = db.session.query(
        CleanVideo.trending_country_code,
        CleanVideo.trending_date,
        CleanVideo.genre,
    ).filter(
        CleanVideo.trending_date.isnot(None),
        CleanVideo.trending_country_code.isnot(None),
        CleanVideo.genre.isnot(None),
    ).all()

    if not rows:
        raise RuntimeError(
            "No data found in videos_clean for visualization. "
            "Make sure preprocessing and loading ran correctly."
        )

    # (region, month) -> {genre: count}
    buckets = defaultdict(lambda: defaultdict(int))

    for region_code, trending_date, genre in rows:
        if not region_code or not trending_date or not genre:
            continue

        month = trending_date.month  # 1–12
        key = (region_code, month)
        buckets[key][genre] += 1

    regions = sorted({region for (region, _m) in buckets.keys()})
    genres = sorted({g for counts in buckets.values() for g in counts.keys()})

    # Convert counts to shares
    shares = {}
    for (region, month), counts in buckets.items():
        total = sum(counts.values()) or 1
        shares[(region, month)] = {
            g: counts.get(g, 0) / total for g in genres
        }

    return regions, genres, shares


def plot_stacked_bar_for_region(region, genres, shares, output_dir: Path):
    """
    Create a stacked bar chart: months on x-axis, stacked genre shares.
    """
    months = list(range(1, 13))

    # Build matrix: rows = genres, cols = months
    data = np.zeros((len(genres), len(months)), dtype=float)

    for col, month in enumerate(months):
        key = (region, month)
        if key in shares:
            for row, g in enumerate(genres):
                data[row, col] = shares[key].get(g, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(months))

    # Stacked bars
    for i, g in enumerate(genres):
        ax.bar(
            months,
            data[i],
            bottom=bottom,
            label=g
        )
        bottom += data[i]

    ax.set_title(f"Genre Share by Month – Region: {region}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Genre Share")
    ax.set_xticks(months)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))

    output_path = output_dir / f"genre_share_stacked_{region}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved stacked bar chart to {output_path}")


def plot_heatmap_for_region(region, genres, shares, output_dir: Path):
    """
    Create a heatmap-like visualization (imshow) for genre share vs month.
    Rows: genres, Columns: months.
    """
    months = list(range(1, 13))
    data = np.zeros((len(genres), len(months)), dtype=float)

    for col, month in enumerate(months):
        key = (region, month)
        if key in shares:
            for row, g in enumerate(genres):
                data[row, col] = shares[key].get(g, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", origin="lower")

    ax.set_title(f"Genre Share Heatmap – Region: {region}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Genre")

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)

    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share (0–1)")

    output_path = output_dir / f"genre_share_heatmap_{region}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved heatmap chart to {output_path}")


def main():
    app = create_app()
    with app.app_context():
        regions, genres, shares = build_region_month_genre_shares()

        print("Available regions:", regions)
        print("Detected genres:", genres)

        target_region = "US"
        if target_region not in regions:
            # Fallback: pick the first region if US is not present
            target_region = regions[0]

        output_dir = Path("scripts/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_stacked_bar_for_region(target_region, genres, shares, output_dir)
        plot_heatmap_for_region(target_region, genres, shares, output_dir)


if __name__ == "__main__":
    main()
