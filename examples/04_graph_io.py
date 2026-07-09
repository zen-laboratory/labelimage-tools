"""Graph construction and graph file I/O example.

Run this file from the repository root with:

    python examples/04_graph_io.py

The script writes graph files and a PNG figure to ``examples/plots``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import labelimage_tools as lit  # noqa: E402

SAMPLE_PATH = REPOSITORY / "samples" / "test_cells2D.tif"
PLOT_DIR = REPOSITORY / "examples" / "plots"
BACKGROUND = 0
SEED = 4
K = 8
DPI = 150


def save_figure(fig, filename: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> list[Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = lit.load_label_image(SAMPLE_PATH)
    labels, _ = lit.crop_to_foreground_bbox(labels, background=BACKGROUND, padding=10)

    neighbors, contacts, centroids, pixel_counts = lit.graph_from_labels(
        labels,
        background=BACKGROUND,
        eight=True,
        include_centroids=True,
        include_pixel_counts=True,
    )

    npz_path = PLOT_DIR / "label_graph.npz"
    json_path = PLOT_DIR / "label_graph.json"
    lit.save_label_graph(
        npz_path,
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
        source_image=str(SAMPLE_PATH),
    )
    lit.save_label_graph(
        json_path,
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
        source_image=str(SAMPLE_PATH),
    )

    loaded_npz = lit.load_label_graph(npz_path)
    loaded_json = lit.load_label_graph(json_path)

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Adjacency graph loaded from graph file",
    )
    lit.draw_graph(
        labels,
        loaded_npz.neighbors,
        contacts=loaded_npz.contacts,
        ax=ax,
        show_labels=False,
        lw_scaling=("sqrt", 0.15),
        line_args={"color": "white", "alpha": 0.75},
    )
    figure_path = save_figure(fig, "adjacency_graph_from_loaded_graph.png")

    print(f"Loaded {SAMPLE_PATH}")
    print(f"Nodes: {len(loaded_npz.neighbors)}")
    print(f"JSON metadata keys: {sorted(loaded_json.metadata)}")
    print("Wrote:")
    for path in [npz_path, json_path, figure_path]:
        print(f"  {path}")
    return [npz_path, json_path, figure_path]


if __name__ == "__main__":
    main()
