"""
src/preparation/class_distribution.py
========================================
Analyzes and visualizes the class distribution in the labeled dataset.

Loads ``data/raw/labels/toi_labels.csv``, prints class counts and percentages,
generates a bar chart saved to ``reports/class_distribution.png``, and prints
summary statistics (period, depth, duration) per class.

Usage
-----
.. code-block:: bash

    python src/preparation/class_distribution.py \\
        --labels data/raw/labels/toi_labels.csv \\
        --output reports/class_distribution.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import get, load_config, project_root  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    logger.warning("matplotlib not installed; plot will be skipped.")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    0: "PLANET",
    1: "ECLIPSING_BINARY",
    2: "BLEND",
    3: "NOISE",
}

# Colour palette (dark-style friendly)
_PALETTE = {
    "PLANET": "#4caf50",
    "ECLIPSING_BINARY": "#f44336",
    "BLEND": "#ff9800",
    "NOISE": "#9e9e9e",
}

STAT_COLS = ["period_days", "depth_ppm", "duration_hrs"]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _print_class_counts(df: pd.DataFrame) -> None:
    """Print class counts and percentages to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Label dataframe with ``label_name`` column.
    """
    total = len(df)
    print("\n" + "=" * 52)
    print(f"{'Class Distribution':^52}")
    print("=" * 52)
    print(f"{'Label':<22} {'Count':>8} {'Percent':>10}")
    print("-" * 52)

    counts = df["label_name"].value_counts()
    for name in LABEL_NAMES.values():
        n = counts.get(name, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {name:<20} {n:>8d} {pct:>9.1f}%")

    print("-" * 52)
    print(f"  {'TOTAL':<20} {total:>8d} {'100.0':>9}%")
    print("=" * 52)


def _print_per_class_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for numerical columns per class.

    Parameters
    ----------
    df : pd.DataFrame
        Label dataframe containing ``label_name`` and stat columns.
    """
    print("\n" + "=" * 78)
    print(f"{'Per-Class Summary Statistics':^78}")
    print("=" * 78)

    present_cols = [c for c in STAT_COLS if c in df.columns]
    if not present_cols:
        print("  No numeric stat columns found.")
        return

    for col in present_cols:
        print(f"\n  [ {col} ]")
        col_header = f"{'Class':<22} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}"
        print("  " + col_header)
        print("  " + "-" * (len(col_header)))
        for name in LABEL_NAMES.values():
            subset = pd.to_numeric(
                df.loc[df["label_name"] == name, col], errors="coerce"
            ).dropna()
            if len(subset) == 0:
                row_str = f"  {name:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            else:
                row_str = (
                    f"  {name:<22}"
                    f" {subset.mean():>10.3f}"
                    f" {subset.median():>10.3f}"
                    f" {subset.std():>10.3f}"
                    f" {subset.min():>10.3f}"
                    f" {subset.max():>10.3f}"
                )
            print(row_str)

    print("=" * 78)


def _generate_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Generate and save a bar chart of class counts.

    Parameters
    ----------
    df : pd.DataFrame
        Label dataframe with ``label_name`` column.
    output_path : Path
        Destination PNG file.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping chart generation.")
        return

    counts = df["label_name"].value_counts()
    labels = list(LABEL_NAMES.values())
    values = [counts.get(ln, 0) for ln in labels]
    colors = [_PALETTE.get(ln, "#607d8b") for ln in labels]

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor("#1e1e2e")

        ax_bar = axes[0]
        bars = ax_bar.bar(labels, values, color=colors, edgecolor="#444466", linewidth=0.8, width=0.6)
        ax_bar.set_title("Class Distribution", fontsize=16, fontweight="bold", color="white", pad=14)
        ax_bar.set_xlabel("Class", fontsize=12, color="#cccccc")
        ax_bar.set_ylabel("Count", fontsize=12, color="#cccccc")
        ax_bar.tick_params(colors="white")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax_bar.spines[spine].set_color("#555566")
        ax_bar.set_facecolor("#2a2a3e")
        ax_bar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        total = sum(values)
        for bar, val in zip(bars, values):
            pct = 100.0 * val / total if total else 0
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=9.5, color="white",
            )

        ax_pie = axes[1]
        wedge_props = {"linewidth": 1.5, "edgecolor": "#1e1e2e"}
        wedges, texts, autotexts = ax_pie.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=wedge_props,
        )
        for text in texts:
            text.set_color("white")
            text.set_fontsize(9)
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(8)
        ax_pie.set_title("Class Proportions", fontsize=14, fontweight="bold", color="white", pad=10)
        ax_pie.set_facecolor("#2a2a3e")

        plt.tight_layout(pad=2.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    logger.info("Bar chart saved to %s", output_path)
    print(f"\nChart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_class_distribution(
    labels_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load labels, print statistics, and generate a bar chart.

    Parameters
    ----------
    labels_path : Path, optional
        Path to ``toi_labels.csv``.  Defaults to
        ``data/raw/labels/toi_labels.csv``.
    output_path : Path, optional
        Path for the output PNG.  Defaults to
        ``reports/class_distribution.png``.

    Returns
    -------
    pd.DataFrame
        The loaded labels dataframe.

    Raises
    ------
    FileNotFoundError
        If the labels file does not exist.
    """
    root = project_root()
    if labels_path is None:
        labels_path = root / get("data.raw_dir", "data/raw") / "labels" / "toi_labels.csv"
    if output_path is None:
        output_path = root / "reports" / "class_distribution.png"

    labels_path = Path(labels_path)
    output_path = Path(output_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path, low_memory=False)
    logger.info("Loaded %d label rows from %s", len(df), labels_path)

    # Ensure label_name column exists
    if "label_name" not in df.columns and "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(3).astype(int)
        df["label_name"] = df["label"].map(LABEL_NAMES).fillna("NOISE")

    _print_class_counts(df)
    _print_per_class_stats(df)
    _generate_bar_chart(df, output_path)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize class distribution in the label dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to toi_labels.csv.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the PNG bar chart.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configs/config.yaml.",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        load_config(_args.config)
    except Exception as _e:
        logger.warning("Could not load config: %s. Using built-in defaults.", _e)

    _labels = Path(_args.labels) if _args.labels else None
    _output = Path(_args.output) if _args.output else None

    _df = analyze_class_distribution(labels_path=_labels, output_path=_output)
    print(f"\nAnalysis complete. Total targets: {len(_df)}")
