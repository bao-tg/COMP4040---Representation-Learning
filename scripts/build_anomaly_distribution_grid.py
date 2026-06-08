from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAYS_DIR = PROJECT_ROOT / "experiments/server_full/anomaly/arrays"
PNG_OUTPUT_PATHS = [
    PROJECT_ROOT / "reports/docs/figures/anomaly_distribution_grid.png",
    PROJECT_ROOT / "reports/final/figures/anomaly_distribution_grid.png",
]
PDF_OUTPUT_PATHS = [
    PROJECT_ROOT / "reports/docs/figures/anomaly_distribution_grid.pdf",
    PROJECT_ROOT / "reports/final/figures/anomaly_distribution_grid.pdf",
]

ENCODERS = [
    ("tfidf", "TF-IDF"),
    ("w2v", "Word2Vec"),
    ("glove", "GloVe"),
    ("sbert", "SBERT"),
    ("bge", "BGE-large"),
]

COLUMNS = [
    ("iso_scores", "Isolation Forest score", "Decision-function score", "#C5D2D7", "#2F5663"),
    ("rating_residuals", "Rating residual", "Absolute residual", "#D8CABD", "#654A3A"),
]

GRID_COLOR = "#E8E6E1"
TEXT_COLOR = "#171717"
SPINE_COLOR = "#5A5A5A"
TITLE_FONT_SIZE = 12.4
TICK_FONT_SIZE = 7.6
SUPTITLE_FONT_SIZE = 13.4


def configure_style() -> None:
    from matplotlib import font_manager

    for font_path in [
        "/mnt/c/Windows/Fonts/times.ttf",
        "/mnt/c/Windows/Fonts/timesbd.ttf",
        "/mnt/c/Windows/Fonts/timesi.ttf",
        "/mnt/c/Windows/Fonts/timesbi.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.unicode_minus": False,
            "pdf.use14corefonts": True,
            "savefig.dpi": 300,
        }
    )


def y_tick_formatter(value: float, _: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def smooth_counts(counts: np.ndarray, radius: float = 1.25) -> np.ndarray:
    window = np.arange(-5, 6, dtype="float64")
    kernel = np.exp(-0.5 * (window / radius) ** 2)
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def array_path(encoder: str, kind: str) -> Path:
    return ARRAYS_DIR / f"{encoder}_{kind}.npy"


def load_values(encoder: str, kind: str) -> np.ndarray:
    path = array_path(encoder, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run the anomaly stage after src/full_pipeline.py saves anomaly arrays."
        )
    return np.load(path, mmap_mode="r")


def plot_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    show_y: bool,
    bar_color: str,
    line_color: str,
) -> None:
    counts, bins = np.histogram(values, bins=80)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    ax.bar(
        centers,
        counts,
        width=widths,
        align="center",
        color=bar_color,
        edgecolor=line_color,
        linewidth=0.18,
        alpha=0.82,
    )
    ax.plot(centers, smooth_counts(counts), color=line_color, linewidth=0.95)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=0.42, length=2.2, pad=1.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(FuncFormatter(y_tick_formatter))

    for spine in ax.spines.values():
        spine.set_linewidth(0.48)
        spine.set_color(SPINE_COLOR)

    if not show_y:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def main() -> None:
    configure_style()

    fig, axes = plt.subplots(
        len(ENCODERS),
        len(COLUMNS),
        figsize=(9.2, 10.6),
        dpi=300,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row, (encoder, encoder_label) in enumerate(ENCODERS):
        for col, (kind, column_title, x_label, bar_color, line_color) in enumerate(COLUMNS):
            values = load_values(encoder, kind)
            ax = axes[row, col]
            plot_distribution(
                ax,
                values,
                show_y=(col == 0),
                bar_color=bar_color,
                line_color=line_color,
            )

            if row == 0:
                ax.set_title(column_title, fontsize=TITLE_FONT_SIZE, fontweight="normal", pad=7)
            if row == len(ENCODERS) - 1:
                ax.set_xlabel(x_label, fontsize=TITLE_FONT_SIZE, labelpad=5)

        left_box = axes[row, 0].get_position()
        fig.text(
            0.083,
            (left_box.y0 + left_box.y1) / 2,
            encoder_label,
            ha="right",
            va="center",
            fontsize=TITLE_FONT_SIZE,
            color=TEXT_COLOR,
        )

    fig.suptitle("Anomaly Distribution Comparison", y=0.965, fontsize=SUPTITLE_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.014, 0.49, "Count", rotation=90, ha="center", va="center", fontsize=TITLE_FONT_SIZE, color=TEXT_COLOR)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.095, top=0.925, wspace=0.18, hspace=0.34)

    for output_path in PNG_OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    for output_path in PDF_OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
