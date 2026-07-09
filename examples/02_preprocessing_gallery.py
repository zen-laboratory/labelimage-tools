"""Preprocessing gallery for label images.

This script demonstrates dilation, erosion, skeletonization, hole filling, and
disconnected-fragment cleanup. It writes PNG files to ``examples/plots``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import labelimage_tools as lit  # noqa: E402

SAMPLE_PATH = REPOSITORY / "samples" / "test_cells2D.tif"
PLOT_DIR = REPOSITORY / "examples" / "plots"
BACKGROUND = 0
SEED = 7
K = 8
DPI = 150


def save_figure(fig, filename: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _show(ax, labels, title: str) -> None:
    lit.plot_label_image(
        labels,
        ax=ax,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title=title,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def synthetic_cleanup_image() -> np.ndarray:
    """Create a tiny image with a hole and a disconnected fragment."""
    labels = np.zeros((60, 100), dtype=np.int64)
    labels[6:54, 6:54] = 5
    labels[16:44, 16:44] = 0
    labels[8:43, 62:97] = 10
    labels[53:56, 95:98] = 10
    return labels


def main() -> list[Path]:
    labels = lit.load_label_image(SAMPLE_PATH)
    labels, _ = lit.crop_to_foreground_bbox(labels, background=BACKGROUND, padding=10)

    eroded = lit.erode_labels(labels, structure=3, background=BACKGROUND)
    dilated = lit.dilate_labels(labels, structure=3, background=BACKGROUND, background_only=True)
    interior_skeleton = lit.skeletonize_labels(labels, background=BACKGROUND, kind="interior")
    exterior_skeleton = lit.skeletonize_labels(labels, background=BACKGROUND, kind="exterior")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    _show(axes[0, 0], labels, "Original labels")
    _show(axes[0, 1], eroded, "Eroded labels")
    _show(axes[0, 2], dilated, "Dilated labels")
    _show(axes[1, 0], interior_skeleton, "Interior skeleton")
    _show(axes[1, 1], exterior_skeleton, "Exterior skeleton")
    shuffled = lit.shuffle_labels(labels, seed=SEED, background=BACKGROUND)
    _show(axes[1, 2], shuffled, "Shuffled label values")
    preprocessing_gallery = save_figure(fig, "preprocessing_gallery.png")

    synthetic = synthetic_cleanup_image()
    filled = lit.fill_internal_gaps_edt(synthetic, background=BACKGROUND, max_distance=5)
    cleaned = lit.remove_non_self_connected_bits(filled, background=BACKGROUND)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    _show(axes[0], synthetic, "Hole + fragment")
    _show(axes[1], filled, "Internal gap filled (max_distance=5)")
    _show(axes[2], cleaned, "Small fragment removed")
    cleanup_gallery = save_figure(fig, "cleanup_gallery.png")

    print("Wrote:")
    for path in [preprocessing_gallery, cleanup_gallery]:
        print(f"  {path}")
    return [preprocessing_gallery, cleanup_gallery]


if __name__ == "__main__":
    main()
